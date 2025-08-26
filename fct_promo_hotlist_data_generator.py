import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

df = pd.read_csv('FCT_PROMO_HOTLIST_TODAY.csv')
# -----------------------------
# Synthetic Retail Time-Series Generator
# -----------------------------
# Design goals:
# - Preserve empirical distributions (bootstrap from your sample 'df').
# - Respect causal relationships:
#   EXPECTED_DAY_UNITS = BASELINE_UNITS * PLANNED_LIFT
#   EXPECTED_SO_FAR_UNITS = EXPECTED_DAY_UNITS * DAY_ELAPSED_SHARE
#   PCT_TO_PLAN_SO_FAR = UNITS_SOLD_SO_FAR / max(EXPECTED_SO_FAR_UNITS, eps)
#   Revenue = price * units (with realistic noise and mismatches).
# - Inventory rolls down with sales; OOS risk triggered when ONHAND_LATEST hits 0 early.
# - Price mismatch probability varies by promo type.
# - Compliance correlates with underperformance and OOS risk.
# - Image URL pattern preserved.

RNG_SEED = 42

# -----------------------------
# Geo Catalog and Helpers
# -----------------------------
# Static, curated city ZIP centroids per region. ZIPs are representative; lat/lon are approximate centroids.
GEO_CATALOG = {
    "Mountain": [
        {"city": "Denver", "state": "CO", "zip": "80202", "lat": 39.7525, "lon": -104.9995},
        {"city": "Aurora", "state": "CO", "zip": "80012", "lat": 39.7076, "lon": -104.8372},
        {"city": "Boulder", "state": "CO", "zip": "80302", "lat": 40.0195, "lon": -105.2927},
        {"city": "FortCollins", "state": "CO", "zip": "80521", "lat": 40.5853, "lon": -105.0844},
        {"city": "ColoradoSprings", "state": "CO", "zip": "80903", "lat": 38.8339, "lon": -104.8214},
        {"city": "Greeley", "state": "CO", "zip": "80631", "lat": 40.4233, "lon": -104.7091},
        {"city": "Arvada", "state": "CO", "zip": "80003", "lat": 39.8240, "lon": -105.0649},
        {"city": "Lakewood", "state": "CO", "zip": "80226", "lat": 39.7108, "lon": -105.0823},
        {"city": "Parker", "state": "CO", "zip": "80134", "lat": 39.5186, "lon": -104.7614},
        {"city": "Thornton", "state": "CO", "zip": "80229", "lat": 39.8650, "lon": -104.9563}
    ],
    "Northeast": [
        {"city": "NewYork", "state": "NY", "zip": "10001", "lat": 40.7506, "lon": -73.9970},
        {"city": "Boston", "state": "MA", "zip": "02108", "lat": 42.3570, "lon": -71.0637},
        {"city": "Philadelphia", "state": "PA", "zip": "19103", "lat": 39.9526, "lon": -75.1652},
        {"city": "Newark", "state": "NJ", "zip": "07102", "lat": 40.7357, "lon": -74.1724},
        {"city": "Hartford", "state": "CT", "zip": "06103", "lat": 41.7658, "lon": -72.6734}
    ],
    "South": [
        {"city": "Dallas", "state": "TX", "zip": "75201", "lat": 32.7876, "lon": -96.7994},
        {"city": "Austin", "state": "TX", "zip": "78701", "lat": 30.2711, "lon": -97.7437},
        {"city": "Atlanta", "state": "GA", "zip": "30303", "lat": 33.7537, "lon": -84.3863},
        {"city": "Miami", "state": "FL", "zip": "33130", "lat": 25.7680, "lon": -80.2044},
        {"city": "Charlotte", "state": "NC", "zip": "28202", "lat": 35.2271, "lon": -80.8431},
        {"city": "Nashville", "state": "TN", "zip": "37203", "lat": 36.1526, "lon": -86.7898}
    ],
    "West": [
        {"city": "LosAngeles", "state": "CA", "zip": "90012", "lat": 34.0617, "lon": -118.2400},
        {"city": "SanFrancisco", "state": "CA", "zip": "94103", "lat": 37.7749, "lon": -122.4194},
        {"city": "Seattle", "state": "WA", "zip": "98104", "lat": 47.6026, "lon": -122.3284},
        {"city": "Portland", "state": "OR", "zip": "97205", "lat": 45.5202, "lon": -122.6742},
        {"city": "LasVegas", "state": "NV", "zip": "89101", "lat": 36.1716, "lon": -115.1391}
    ]
}

def _extract_city_from_location_name(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    # Try to match known city tokens present in GEO_CATALOG
    tokens = name.replace("#", " ").replace("-", " ").replace(",", " ").split()
    lowered = [t.lower() for t in tokens]
    all_cities = {entry["city"].lower(): entry["city"] for region in GEO_CATALOG.values() for entry in region}
    for lc, canon in all_cities.items():
        parts = [lc]
        # also allow spaced variants e.g., FortCollins -> Fort + Collins
        if lc == "fortcollins":
            parts.append("fort")
            parts.append("collins")
        if lc == "coloradosprings":
            parts.append("colorado")
            parts.append("springs")
        # If any token matches start-of-word sequence
        if lc in lowered:
            return canon
        if all(p in lowered for p in parts if p not in (lc,)):
            return canon
    return None

def _choose_geo_for_store(region: str, location_name: str, location_id: str) -> Tuple[str, str, float, float]:
    region_key = region if region in GEO_CATALOG else np.random.choice(list(GEO_CATALOG.keys()))
    candidates = GEO_CATALOG[region_key]
    # Prefer city inferred from LOCATION_NAME
    inferred_city = _extract_city_from_location_name(location_name)
    if inferred_city is not None:
        for c in candidates:
            if c["city"] == inferred_city:
                return c["state"], c["zip"], float(c["lat"]), float(c["lon"])
    # Deterministic pick by LOCATION_ID hash
    rng = np.random.default_rng(abs(hash(location_id)) % (2**32))
    chosen = candidates[int(rng.integers(low=0, high=len(candidates)))]
    return chosen["state"], chosen["zip"], float(chosen["lat"]), float(chosen["lon"])

# Preferred legacy Soft Drink item names to preserve
LIKED_SOFT_DRINK_NAMES = [
    "CanyonCola Energy 20oz - Peach",
    "VoltRush Energy 16oz - Citrus",
    "CanyonCola Energy 20oz - Tropical",
    "VoltRush Energy 12oz - Tropical",
    "CanyonCola Energy 12oz - Berry",
    "CanyonCola Energy 16oz - Lime",
    "HydraBolt Energy 20oz - Citrus",
    "ZenFizz Energy 12oz - Citrus",
    "CanyonCola Energy 12oz - Lime",
    "ZenFizz Energy 16oz - Tropical"
]

# Image directories (absolute paths)
OPEN_IMG_DIR = "/Users/cassidy.hilton/Cursor Projects/unitFocus/img/open"
MIDDAY_IMG_DIR = "/Users/cassidy.hilton/Cursor Projects/unitFocus/img/midday"

def _scan_images(img_dir: str) -> Dict[str, str]:
    """Scan a directory and return a mapping: base filename (no ext) -> absolute path."""
    mapping: Dict[str, str] = {}
    try:
        if os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    key = os.path.splitext(fname)[0]
                    mapping[key] = os.path.join(img_dir, fname)
    except Exception:
        pass
    return mapping

# Preload image maps once
OPEN_IMAGE_MAP = _scan_images(OPEN_IMG_DIR)
MIDDAY_IMAGE_MAP = _scan_images(MIDDAY_IMG_DIR)

def _extract_brand(item_name: str) -> str:
    parts = item_name.split(" ")
    return parts[0] if parts else ""

def _resolve_image_path(item_name: str, when: str) -> str:
    """
    Resolve best image path for item_name.
    Priority:
      1) Exact match in requested folder (open/midday)
      2) Exact match in the other folder
      3) Brand-level match in requested folder
      4) Brand-level match in the other folder
      5) First available image in requested folder
      6) First available image in the other folder
      7) Empty string if none found
    """
    open_map = OPEN_IMAGE_MAP
    mid_map = MIDDAY_IMAGE_MAP
    brand = _extract_brand(item_name)

    def first_value(d: Dict[str, str]) -> Optional[str]:
        return next(iter(sorted(d.values()))) if d else None

    # Specialty Beverages: prefer a standardized 16oz image key for a given brand and flavor,
    # regardless of the size listed in ITEM_NAME. Fallback to other sizes if needed.
    def _standardize_energy_key(name: str) -> tuple:
        """Return (brand, variant, preferred_size) for specialty energy items, else (None,None,None)."""
        try:
            if "Energy" in name and any(name.startswith(b + " ") for b in SPECIALTY_BEVERAGE_BRANDS):
                left, variant = name.split(" - ", 1)
                brand = left.split(" ")[0]
                # Special rule: CanyonCola Peach -> 20oz canonical image
                if brand == "CanyonCola" and "Peach" in variant:
                    return brand, "Peach", "20oz"
                # Special rule: ZenFizz Citrus -> 12oz canonical image
                if brand == "ZenFizz" and "Citrus" in variant:
                    return brand, "Citrus", "12oz"
                return brand, variant, "16oz"
        except Exception:
            return None, None, None
        return None, None, None

    def _try_sizes(brand: str, variant: str, target_map: dict, alt_map: dict, preferred_size: str = "16oz") -> str:
        size_order = [preferred_size] + [s for s in ["16oz", "12oz", "20oz"] if s != preferred_size]
        for size in size_order:
            key = f"{brand} Energy {size} - {variant}"
            if key in target_map:
                return target_map[key]
            if key in alt_map:
                # fallback to alt map if not found in target
                return alt_map[key]
        return ""

    target_map = open_map if when == "open" else mid_map
    alt_map = mid_map if when == "open" else open_map

    # 0) Specialty beverage standardization
    brand_std, variant_std, pref_size = _standardize_energy_key(item_name)
    if brand_std is not None:
        std_key = f"{brand_std} Energy {pref_size} - {variant_std}"
        if std_key in target_map:
            return target_map[std_key]
        if std_key in alt_map:
            return alt_map[std_key]
        found = _try_sizes(brand_std, variant_std, target_map, alt_map, preferred_size=pref_size)
        if found:
            return found

    if when == "open":
        if item_name in open_map:
            return open_map[item_name]
        if item_name in mid_map:
            return mid_map[item_name]
        brand_matches = {k: v for k, v in open_map.items() if k.startswith(brand + " ")}
        if brand_matches:
            return first_value(brand_matches) or ""
        brand_matches = {k: v for k, v in mid_map.items() if k.startswith(brand + " ")}
        if brand_matches:
            return first_value(brand_matches) or ""
        return first_value(open_map) or first_value(mid_map) or ""
    else:
        if item_name in mid_map:
            return mid_map[item_name]
        if item_name in open_map:
            return open_map[item_name]
        brand_matches = {k: v for k, v in mid_map.items() if k.startswith(brand + " ")}
        if brand_matches:
            return first_value(brand_matches) or ""
        brand_matches = {k: v for k, v in open_map.items() if k.startswith(brand + " ")}
        if brand_matches:
            return first_value(brand_matches) or ""
        return first_value(mid_map) or first_value(open_map) or ""

# Brands that should be treated as Specialty Beverages when paired with Energy items
SPECIALTY_BEVERAGE_BRANDS = ["VoltRush", "HydraBolt", "ZenFizz", "PowerMax", "CanyonCola"]

def _normalize_category_for_item(item_name: str, category: str) -> str:
    try:
        if "Energy" in item_name and any(item_name.startswith(b) for b in SPECIALTY_BEVERAGE_BRANDS):
            return "Specialty Beverages"
    except Exception:
        pass
    return category

def _bootstrap_choices(series: pd.Series, size: int) -> np.ndarray:
    """Empirical sampler with replacement for categorical-like fields."""
    vals = series.dropna().values
    if len(vals) == 0:
        return np.array([np.nan] * size)
    rng = np.random.default_rng(RNG_SEED)
    return rng.choice(vals, size=size, replace=True)

def _fit_price_noise(df: pd.DataFrame) -> Tuple[float, float]:
    """Estimate price noise (observed - plan) dispersion from sample."""
    diffs = (df["OBSERVED_PRICE"] - df["PRICE_PLAN"]).dropna().values
    if len(diffs) < 2:
        return 0.0, 0.05  # small default stdev
    return float(np.mean(diffs)), float(max(np.std(diffs), 0.03))

def _promo_lift_empirical(df: pd.DataFrame) -> Dict[str, float]:
    """Median lift by promo type to stay realistic."""
    lifts = (
        df.groupby("PROMO_TYPE")["PLANNED_LIFT"]
          .median()
          .to_dict()
    )
    if not lifts:
        lifts = {"Base": 1.0, "TPR": 1.2, "Display": 1.3, "TPR+Display": 1.45}
    # Ensure common promo types exist
    defaults = {"TPR": 1.25, "Display": 1.30, "TPR+Display": 1.45}
    for k, v in defaults.items():
        lifts.setdefault(k, v)
    return lifts

def _price_mismatch_prob(promo_type: str) -> float:
    """Heuristic probability of price mismatch by promo type."""
    base = {
        "TPR": 0.12,
        "Display": 0.06,
        "TPR+Display": 0.10
    }
    return base.get(promo_type, 0.04)  # default small chance

def _compliance_label(underperforming: int, oos_risk: int) -> str:
    """Correlate compliance with risk/underperformance."""
    if oos_risk or underperforming:
        # 65% chance of NON_COMPLIANT if issues exist
        return "NON_COMPLIANT" if np.random.random() < 0.65 else "COMPLIANT"
    # Otherwise mostly compliant
    return "COMPLIANT" if np.random.random() < 0.85 else "NON_COMPLIANT"

def _root_cause(oos_risk: int, price_mismatch_flag: int) -> Optional[str]:
    if oos_risk:
        return np.random.choice([
            "OOS/Backroom not worked",
            "Late Replenishment",
            "Phantom Inventory"
        ], p=[0.5, 0.3, 0.2])
    if price_mismatch_flag:
        return np.random.choice([
            "Price Label Error",
            "Incorrect POS Promo",
            "Promo Not Loaded"
        ], p=[0.6, 0.25, 0.15])
    # Occasionally assign other causes
    if np.random.random() < 0.08:
        return np.random.choice([
            "Display Not Built",
            "Poor Shelf Placement",
            "Competing Promo Nearby"
        ])
    return np.nan

def _make_catalog_from_sample(df: pd.DataFrame,
                              n_stores: int,
                              n_skus: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create store and product catalogs by bootstrapping from sample + adding new IDs."""
    rng = np.random.default_rng(RNG_SEED)

    # Derive base pools
    store_ids = df["LOCATION_ID"].dropna().unique().tolist()
    store_names = df.drop_duplicates("LOCATION_ID")[["LOCATION_ID", "LOCATION_NAME"]].set_index("LOCATION_ID")["LOCATION_NAME"].to_dict()
    regions = df.drop_duplicates("LOCATION_ID")[["LOCATION_ID", "REGION"]].set_index("LOCATION_ID")["REGION"].to_dict()

    # If too few, synthesize more
    def _synthesize_store(i: int):
        sid = f"TGT{1001 + i:04d}"
        city = np.random.choice(["Denver", "Aurora", "Parker", "Thornton", "Greeley", "Boulder", "Arvada", "Lakewood", "FortCollins", "ColoradoSprings"])
        name = f"Target {city} #{1001 + i}"
        region = np.random.choice(["Mountain", "Northeast", "South", "West"])
        return sid, name, region

    # Build target number of stores
    stores = []
    i = 0
    while len(stores) < n_stores:
        if i < len(store_ids):
            sid = store_ids[i]
            stores.append((
                sid,
                store_names.get(sid, f"Target #{sid[3:]}"),
                regions.get(sid, np.random.choice(["Mountain", "Northeast", "South", "West"]))
            ))
        else:
            stores.append(_synthesize_store(i))
        i += 1
    store_df = pd.DataFrame(stores, columns=["LOCATION_ID", "LOCATION_NAME", "REGION"])
    # Enrich with geo fields per LOCATION_ID
    states, zips, lats, lons, cities = [], [], [], [], []
    for _, row in store_df.iterrows():
        st, zp, la, lo = _choose_geo_for_store(str(row["REGION"]), str(row["LOCATION_NAME"]), str(row["LOCATION_ID"]))
        states.append(st)
        zips.append(zp)
        lats.append(la)
        lons.append(lo)
        # derive city from centroid match
        region_key = row["REGION"] if row["REGION"] in GEO_CATALOG else np.random.choice(list(GEO_CATALOG.keys()))
        matched_city = None
        for cand in GEO_CATALOG[region_key]:
            if cand["zip"] == zp:
                matched_city = cand["city"]
                break
        cities.append(matched_city or "Unknown")
    store_df["STATE"] = states
    store_df["ZIP_CODE"] = zips
    store_df["LATITUDE"] = lats
    store_df["LONGITUDE"] = lons
    store_df["CITY"] = cities

    # Products
    sku_ids = df["ITEM_ID"].dropna().unique().tolist()
    sku_names = df.drop_duplicates("ITEM_ID")[["ITEM_ID", "ITEM_NAME"]].set_index("ITEM_ID")["ITEM_NAME"].to_dict()
    categories = df.drop_duplicates("ITEM_ID")[["ITEM_ID", "CATEGORY"]].set_index("ITEM_ID")["CATEGORY"].to_dict()

    def _synthesize_sku(i: int):
        sid = f"SKU{90000 + i:05d}"
        
        # Define product categories with aligned brands and product types
        product_templates = {
            "Specialty Beverages": {
                "brands": ["VoltRush", "HydraBolt", "ZenFizz", "PowerMax", "CanyonCola"],
                "variants": ["Citrus", "Berry", "Peach", "Lime", "Tropical", "Original"],
                "sizes": ["12oz", "16oz", "20oz"],
                "product_type": "Energy"
            },
            "Beverage": {
                "brands": ["ClassicCola", "MountainMist", "CitrusBlast", "SodaSpark"],
                "variants": ["Original", "Diet", "Zero", "Cherry", "Vanilla"],
                "sizes": ["12oz", "16oz", "20oz", "2L"],
                "product_type": ""
            },
            "Snacks": {
                "brands": ["CrunchTime", "GoldenCrisp", "FlavorWave", "SnackMaster"],
                "variants": ["Original", "BBQ", "Sour Cream", "Spicy", "Cheese"],
                "sizes": ["1oz", "2.5oz", "5oz", "Family Size"],
                "product_type": "Chips"
            },
            "Health & Wellness": {
                "brands": ["VitalLife", "WellnessPlus", "NutriBoost", "HealthyChoice"],
                "variants": ["Multivitamin", "Vitamin D", "Omega-3", "Probiotics", "Calcium"],
                "sizes": ["30ct", "60ct", "90ct", "120ct"],
                "product_type": "Supplement"
            },
            "Personal Care": {
                "brands": ["FreshStart", "CleanCare", "PurePlus", "EssentialCare"],
                "variants": ["Original", "Sensitive", "Whitening", "Mint", "Unscented"],
                "sizes": ["Travel", "Regular", "Family", "Value"],
                "product_type": "Toothpaste"
            },
            "Candy & Confectionery": {
                "brands": ["SweetTreat", "CandyLand", "TastyBite", "SugarRush"],
                "variants": ["Original", "Sour", "Fruit", "Chocolate", "Mixed"],
                "sizes": ["Fun Size", "Regular", "King Size", "Share Size"],
                "product_type": "Gummies"
            },
            "Dairy & Eggs": {
                "brands": ["FarmFresh", "DairyPure", "GoldenHen", "Creamline"],
                "variants": ["Whole", "2%", "1%", "Skim", "Cage-Free"],
                "sizes": ["12ct", "18ct", "Half Gallon", "Gallon"],
                "product_type": "Milk"
            },
            "Bakery": {
                "brands": ["BakeHouse", "DailyBread", "Crust & Crumb", "MorningRise"],
                "variants": ["White", "Whole Wheat", "Sourdough", "Multigrain", "Cinnamon"],
                "sizes": ["Loaf", "6ct", "8ct", "12ct"],
                "product_type": "Bread"
            },
            "Frozen Foods": {
                "brands": ["FrostBite", "ArcticDelight", "QuickHeat", "DeepFreeze"],
                "variants": ["Pepperoni", "Cheese", "Veggie", "Chicken", "Mixed Veg"],
                "sizes": ["12oz", "24oz", "Family Size"],
                "product_type": "Pizza"
            },
            "Household Cleaning": {
                "brands": ["PureHome", "ShinePro", "CleanMaster", "Sparkle"],
                "variants": ["Fresh", "Lemon", "Lavender", "Unscented"],
                "sizes": ["32oz", "64oz", "Value Pack"],
                "product_type": "All-Purpose Cleaner"
            },
            "Pet Care": {
                "brands": ["PawPrime", "HappyPup", "WhiskerWell", "TailTreat"],
                "variants": ["Chicken", "Beef", "Salmon", "Turkey"],
                "sizes": ["3lb", "8lb", "16lb"],
                "product_type": "Dog Food"
            },
            "Baby Care": {
                "brands": ["LittleSteps", "GentleCare", "TinyTots", "SnugFit"],
                "variants": ["Newborn", "Size 1", "Size 2", "Sensitive"],
                "sizes": ["24ct", "36ct", "72ct"],
                "product_type": "Diapers"
            },
            "Breakfast & Cereal": {
                "brands": ["SunCrunch", "MorningFlakes", "HoneyOats", "FiberMax"],
                "variants": ["Honey", "Cinnamon", "Original", "Protein"],
                "sizes": ["10oz", "18oz", "Family Size"],
                "product_type": "Cereal"
            },
            "Pantry": {
                "brands": ["PastaPrima", "RedSauce Co.", "GoldenGrain", "ChefSelect"],
                "variants": ["Spaghetti", "Penne", "Marinara", "Alfredo"],
                "sizes": ["16oz", "24oz", "32oz"],
                "product_type": "Pasta"
            },
            "Produce": {
                "brands": ["FieldFresh", "OrchardBest", "GardenPick", "GreenLeaf"],
                "variants": ["Apples", "Bananas", "Salad Mix", "Berries"],
                "sizes": ["1lb", "2lb", "3lb", "Family Pack"],
                "product_type": "Fresh"
            },
            "Meat & Seafood": {
                "brands": ["Butcher's Choice", "SeaHarvest", "FarmToTable", "PrimeCuts"],
                "variants": ["Chicken Breast", "Ground Beef", "Salmon", "Shrimp"],
                "sizes": ["1lb", "2lb", "Family Pack"],
                "product_type": ""
            }
        }
        
        # Randomly select a category and build appropriate product
        category = np.random.choice(list(product_templates.keys()))
        template = product_templates[category]
        
        brand = np.random.choice(template["brands"])
        variant = np.random.choice(template["variants"])
        size = np.random.choice(template["sizes"])
        product_type = template["product_type"]
        
        if product_type:
            if category == "Beverage":
                name = f"{brand} {variant} {size}"
            else:
                name = f"{brand} {product_type} {size} - {variant}"
        else:
            name = f"{brand} {variant} {size}"
            
        # Normalize category for specialty energy beverages
        category = _normalize_category_for_item(name, category)
        return sid, name, category

    prods = []
    # Ensure preferred Soft Drink item names are present
    forced_i = 0
    for liked_name in LIKED_SOFT_DRINK_NAMES:
        if len(prods) >= n_skus:
            break
        pid = f"SKU{98000 + forced_i:05d}"
        prods.append((pid, liked_name, "Specialty Beverages"))
        forced_i += 1

    i = 0
    while len(prods) < n_skus:
        if i < len(sku_ids):
            pid = sku_ids[i]
            name_guess = sku_names.get(pid, f"Product {pid}")
            cat_guess = categories.get(pid, np.random.choice([
                "Specialty Beverages", "Beverage", "Snacks", "Health & Wellness", "Personal Care", "Candy & Confectionery",
                "Dairy & Eggs", "Bakery", "Frozen Foods", "Household Cleaning", "Pet Care", "Baby Care",
                "Breakfast & Cereal", "Pantry", "Produce", "Meat & Seafood"
            ]))
            cat_guess = _normalize_category_for_item(name_guess, cat_guess)
            prods.append((pid, name_guess, cat_guess))
        else:
            prods.append(_synthesize_sku(i))
        i += 1

    sku_df = pd.DataFrame(prods, columns=["ITEM_ID", "ITEM_NAME", "CATEGORY"])
    return store_df, sku_df

def _simulate_one_day(store_row, sku_row, date_dt,
                      price_plan_mu_sigma: Tuple[float,float],
                      promo_lifts: Dict[str, float],
                      baseline_mu_sigma: Tuple[float,float],
                      snapshots_per_day: int = 9) -> Dict:
    """
    Generate one (store, sku, date) observation consistent with your schema.
    """
    rng = np.random.default_rng()

    # PROMO TYPE draw with empirical bias (favor TPR/TPR+Display if present in sample)
    promo_type = np.random.choice(["TPR", "Display", "TPR+Display", "Base"], p=[0.42, 0.18, 0.28, 0.12])

    # Planned lift by promo
    planned_lift = promo_lifts.get(promo_type, 1.0)

    # Baseline units (store- and sku-level): lognormal to avoid negatives
    base_mu, base_sigma = baseline_mu_sigma
    baseline_units = max(1, int(np.random.lognormal(mean=base_mu, sigma=base_sigma)))

    expected_day_units = baseline_units * planned_lift

    # Price plan & observed price
    price_plan = float(np.clip(np.random.normal(loc=price_plan_mu_sigma[0], scale=price_plan_mu_sigma[1]), 1.5, 4.5))

    # Observed price: sometimes mismatched
    mismatch_p = _price_mismatch_prob(promo_type)
    price_mismatch_flag = 1 if np.random.random() < mismatch_p else 0

    # If mismatch, deviate more; else small noise
    noise_mu, noise_sigma = 0.0, 0.05 if not price_mismatch_flag else 0.25
    observed_price = float(np.clip(price_plan + np.random.normal(noise_mu, noise_sigma), 1.0, 5.0))
    price_mismatch_diff = round(observed_price - price_plan, 2) if price_mismatch_flag else 0.0

    # Day progression
    snapshots_seen = snapshots_per_day
    day_elapsed_share = 1.0  # end-of-day snapshot in your sample
    expected_so_far = expected_day_units * day_elapsed_share

    # Realized demand: center on expectation with promo-price elasticity + noise
    price_delta_pct = (observed_price - price_plan) / max(price_plan, 1e-6)
    elasticity = np.random.uniform(0.15, 0.35)  # simple elasticity magnitude
    demand_multiplier = 1.0 - elasticity * price_delta_pct
    demand_multiplier = float(np.clip(demand_multiplier, 0.4, 1.6))
    mu_units = expected_so_far * demand_multiplier
    sigma_units = max(1.0, 0.12 * mu_units)
    units_sold_so_far = int(np.clip(np.random.normal(mu_units, sigma_units), 0, mu_units * 1.8))

    # Inventory: start sized to expectation with some cushion
    onhand_start = int(np.clip(np.random.normal(expected_day_units * 3.0, expected_day_units * 0.8), 40, 250))
    backroom_start = int(np.clip(np.random.normal(onhand_start * 0.45, onhand_start * 0.15), 0, onhand_start))

    # Onhand depletion with possibility of poor replenishment
    work_rate = np.random.uniform(0.55, 0.95)
    shelf_capacity = onhand_start + int(backroom_start * work_rate)
    onhand_latest = max(0, shelf_capacity - units_sold_so_far)

    # Risk heuristic (slightly more sensitive)
    # - Immediate risk if shelf empty while demand expected
    # - Definite risk if very low on hand relative to expected demand
    # - Marginal low stock triggers probabilistic risk
    if (onhand_latest == 0 and expected_day_units > units_sold_so_far * 0.8):
        oos_risk_flag = 1
    elif (onhand_latest <= 2 and expected_day_units > units_sold_so_far):
        oos_risk_flag = 1
    elif (onhand_latest <= 3 and expected_day_units > units_sold_so_far * 0.9 and np.random.random() < 0.35):
        oos_risk_flag = 1
    else:
        oos_risk_flag = 0

    # KPI math
    expected_units_per_snapshot = expected_day_units / snapshots_seen
    units_per_snapshot = (units_sold_so_far / snapshots_seen) if snapshots_seen else 0.0
    trending_gap_units = round(expected_so_far - units_sold_so_far, 2)

    pct_to_plan_so_far = float(units_sold_so_far / max(expected_so_far, 1e-6))
    revenue_so_far = round(observed_price * units_sold_so_far, 2)
    expected_revenue_so_far = round(price_plan * expected_so_far, 3)
    revenue_gap_so_far = round(expected_revenue_so_far - revenue_so_far, 3)

    # -------- NEW: Forecasted revenue (pace-adjusted to full day) --------
    eps = 1e-6
    revenue_forecasted = round(revenue_so_far / max(day_elapsed_share, eps), 2)
    # ---------------------------------------------------------------------

    # Flags & priority score (higher when underperforming or OOS risk or price mismatch)
    underperforming_flag = 1 if pct_to_plan_so_far < 0.9 else 0
    priority_score = (
        (revenue_gap_so_far if revenue_gap_so_far > 0 else 0) +
        (35 if oos_risk_flag else 0) +
        (18 if price_mismatch_flag else 0) +
        max(0, 5 * (0.9 - pct_to_plan_so_far))
    )

    # Compliance label reflects issues
    midday_label = _compliance_label(underperforming_flag, oos_risk_flag)

    # Root cause inference
    root_cause_code = _root_cause(oos_risk_flag, price_mismatch_flag)

    # Resolve local image paths, best-effort
    opening_url = _resolve_image_path(sku_row.ITEM_NAME, "open")
    midday_url = _resolve_image_path(sku_row.ITEM_NAME, "midday")

    return {
        "ITEM_ID": sku_row.ITEM_ID,
        "ITEM_NAME": sku_row.ITEM_NAME,
        "CATEGORY": sku_row.CATEGORY,
        "LOCATION_ID": store_row.LOCATION_ID,
        "LOCATION_NAME": store_row.LOCATION_NAME,
        "REGION": store_row.REGION,
        "STATE": store_row.STATE,
        "ZIP_CODE": store_row.ZIP_CODE,
        "LATITUDE": float(store_row.LATITUDE),
        "LONGITUDE": float(store_row.LONGITUDE),
        "CITY": store_row.CITY,
        "PROMO_TYPE": promo_type,
        "PLANNED_LIFT": round(planned_lift, 2),
        "PRICE_PLAN": round(price_plan, 2),
        "OBSERVED_PRICE": round(observed_price, 2),
        "BASELINE_UNITS": int(baseline_units),
        "EXPECTED_DAY_UNITS": float(round(expected_day_units, 2)),
        "SNAPSHOTS_SEEN": int(9),
        "DAY_ELAPSED_SHARE": float(1.0),
        "UNITS_SOLD_SO_FAR": int(units_sold_so_far),
        "EXPECTED_SO_FAR_UNITS": float(round(expected_so_far, 2)),
        "TRENDING_GAP_UNITS": float(trending_gap_units),
        "ONHAND_START": int(onhand_start),
        "BACKROOM_START": int(backroom_start),
        "ONHAND_LATEST": int(onhand_latest),
        "OSA_FLAG": int(1),
        "ROOT_CAUSE_CODE": root_cause_code,
        "PRICE_MISMATCH_FLAG": int(price_mismatch_flag),
        "PRICE_MISMATCH_DIFF": float(round(price_mismatch_diff, 2)),
        "REVENUE_SO_FAR_USD": float(revenue_so_far),
        "EXPECTED_REVENUE_SO_FAR_USD": float(expected_revenue_so_far),
        "REVENUE_GAP_SO_FAR_USD": float(revenue_gap_so_far),
        "REVENUE_FORECASTED_USD": float(revenue_forecasted),  # <-- NEW FIELD
        "PCT_TO_PLAN_SO_FAR": float(round(pct_to_plan_so_far, 6)),
        "UNITS_PER_SNAPSHOT": float(round(units_per_snapshot, 6)),
        "EXPECTED_UNITS_PER_SNAPSHOT": float(round(expected_units_per_snapshot, 12)),
        "REPLENISH_NOW_FLAG": int(0),   # optional: compute if onhand_latest below threshold
        "STALLED_DISPLAY_FLAG": int(0), # optional: could tie to NON_COMPLIANT + low sales
        "PRIORITY_SCORE": float(round(priority_score, 3)),
        "UNDERPERFORMING_FLAG": int(underperforming_flag),
        "OOS_RISK_FLAG": int(oos_risk_flag),
        "OPENING_IMAGE_URL": opening_url,
        "MIDDAY_IMAGE_URL": midday_url,
        "MIDDAY_COMPLIANCE_LABEL": midday_label,
        "LAST_SNAPSHOT_AT": pd.Timestamp(date_dt.replace(hour=16, minute=0, second=0))
    }

def synthesize_time_series(
    df_sample: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    months: int = 12,
    forecast_horizon_days: int = 90,
    n_stores: int = 50,
    n_skus: int = 25,
    target_rows: int = 10_000
) -> pd.DataFrame:
    """
    Expand the sample df into a realistic time series.

    Date control:
      - If start_date/end_date are None, generate a daily range covering the last `months`
        up to today, then extend forward `forecast_horizon_days`.
      - If end_date is provided, forecast_horizon_days extends beyond it.
      - If both start_date and end_date are provided, those bounds override.
    """

    np.random.seed(RNG_SEED)
    pd.options.mode.chained_assignment = None

    # Resolve date bounds
    if end_date is None:
        today_dt = pd.Timestamp.today().normalize()
        # Full `months`-month historical window ending today
        start_dt = (today_dt - pd.DateOffset(months=months)).normalize() + pd.Timedelta(days=1)
        # Max date = today + 60 days
        end_dt = today_dt + pd.Timedelta(days=60)
    else:
        end_dt = pd.Timestamp(end_date)
        if start_date is None:
            start_dt = (end_dt - pd.DateOffset(months=max(months, 1))).normalize() + pd.Timedelta(days=1)
        else:
            start_dt = pd.Timestamp(start_date)

    # Build date range (inclusive of future forecast window)
    dates = pd.date_range(start=start_dt, end=end_dt, freq="D")

    # Build catalogs from your sample (preserves names/regions/categories)
    store_df, sku_df = _make_catalog_from_sample(df_sample, n_stores=n_stores, n_skus=n_skus)

    # Fit price distribution (plan) from sample
    if "PRICE_PLAN" in df_sample and df_sample["PRICE_PLAN"].notna().any():
        plan_mu = float(df_sample["PRICE_PLAN"].mean())
        plan_sigma = float(max(df_sample["PRICE_PLAN"].std(ddof=0), 0.12))
    else:
        plan_mu, plan_sigma = 2.45, 0.25

    # Baseline units log-normal params from sample
    if "BASELINE_UNITS" in df_sample and df_sample["BASELINE_UNITS"].notna().any():
        bu = df_sample["BASELINE_UNITS"].clip(lower=1).astype(float)
        ln = np.log(bu)
        base_mu, base_sigma = float(np.median(ln)), float(max(np.std(ln), 0.35))
    else:
        base_mu, base_sigma = np.log(30), 0.45

    promo_lifts = _promo_lift_empirical(df_sample)

    # Build full grid; sample down if needed
    total_grid = len(dates) * len(store_df) * len(sku_df)
    combos = [(d, s, p) for d in dates for s in range(len(store_df)) for p in range(len(sku_df))]

    if total_grid > target_rows * 1.15:
        rng = np.random.default_rng(RNG_SEED)
        chosen_idx = rng.choice(len(combos), size=target_rows, replace=False)
        combos = [combos[i] for i in chosen_idx]

    # Simulate
    rows = []
    for d, s_idx, p_idx in combos:
        store_row = store_df.iloc[s_idx]
        sku_row = sku_df.iloc[p_idx]
        rec = _simulate_one_day(
            store_row,
            sku_row,
            pd.Timestamp(d).to_pydatetime(),
            price_plan_mu_sigma=(plan_mu, plan_sigma),
            promo_lifts=promo_lifts,
            baseline_mu_sigma=(base_mu, base_sigma),
            snapshots_per_day=9
        )
        rows.append(rec)

    out = pd.DataFrame(rows)

    # Types and rounding
    float_cols_2dp = [
        "PRICE_PLAN", "OBSERVED_PRICE", "EXPECTED_DAY_UNITS",
        "EXPECTED_SO_FAR_UNITS", "REVENUE_SO_FAR_USD",
        "EXPECTED_REVENUE_SO_FAR_USD", "REVENUE_GAP_SO_FAR_USD",
        "TRENDING_GAP_UNITS", "REVENUE_FORECASTED_USD"
    ]
    for c in float_cols_2dp:
        if c in out.columns:
            out[c] = out[c].astype(float)

    int_cols = [
        "BASELINE_UNITS","UNITS_SOLD_SO_FAR","SNAPSHOTS_SEEN",
        "ONHAND_START","BACKROOM_START","ONHAND_LATEST",
        "OSA_FLAG","PRICE_MISMATCH_FLAG","REPLENISH_NOW_FLAG",
        "STALLED_DISPLAY_FLAG","UNDERPERFORMING_FLAG","OOS_RISK_FLAG"
    ]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].astype(int)

    return out.sort_values(["LAST_SNAPSHOT_AT","LOCATION_ID","ITEM_ID"]).reset_index(drop=True)

df_big = synthesize_time_series(
    df_sample=df,
    months=6,               # 6 months history
    forecast_horizon_days=60, # 60 days into the future
    n_stores=50,
    n_skus=40,
    target_rows=10_000
)

print(df_big["LAST_SNAPSHOT_AT"].min(), df_big["LAST_SNAPSHOT_AT"].max(), df_big.shape)
print(f"...writing FCT_PROMO_HOTLIST_FULL.parquet to Users/cassidy.hilton/Cursor Projects/unitFocus/data")
df_big.to_parquet('FCT_PROMO_HOTLIST_FULL.parquet')
df_big.to_parquet('/Users/cassidy.hilton/Cursor Projects/unitFocus/data/FCT_PROMO_HOTLIST_FULL.parquet')
