from collections import defaultdict
from collections.abc import Iterable, Mapping

EXCLUDE_AMENITIES = {
    # Money / vending / automatons
    "atm",
    "cash_withdrawal",
    "vending_machine",
    "payment_terminal",

    # Sanitary / hygiene micro-POIs
    "toilets",
    "shower",
    "changing_room",
    "drinking_water",
    "dog_toilet",

    # Bicycle / mobility micro-POIs
    "bicycle_parking",
    "bicycle_repair_station",
    "charging_station",
    "taxi",              # a taxi stand is not a building land-use
    "motorcycle_parking",
    "parking",           # generic parking area
    "parking_entrance",
    "parking_space",

    # Street furniture
    "bench",
    "waste_basket",
    "litter_bin",
    "grit_bin",
    "recycling",
    "shelter",
    "fountain",
    "clock",
    "post_box",
    "telephone",
    "emergency_phone",

    # Small public utilities
    "water_point",
    "watering_place",
    "watering_hole",
    "ice_cream",
    "parcel_locker",
    "loading_dock",

    # Electric / network cabinets etc.
    "power_supply",
    "power_outlet",
    "power_box",
    "generator",
    "transformer",
    "substation",

    # Transport infrastructure (not land use)
    "bus_stop",
    "boat_rental",
    "cycle_hire",
    "car_pool",
    "motorcycle_rental",

    # First aid but not a building land-use
    "defibrillator",
    "first_aid_kit",

    # Security / access control
    "gate",
    "security_cage",
    "security_post",

    # Misc public objects
    "bbq",
    "picnic_table",
    "waste_disposal",
    "compressed_air",
    "internet_cafe",  # sometimes room-level not building-level
    "smoking_area",
    "mailroom",

    # Religious micro-POIs (not the building)
    "bible_box",
    "holy_water",
    "prayer_room",

    # Information / signage
    "information",
    "map",
    "tourist_map",
    "board",
    "notice_board",
    "public_bookcase",
    "guidepost",
    "route_marker",

    # Waste / disposal
    "waste_transfer_station",
    "waste_packaging",
    "waste_dump",
    "waste_container",
}

BUILDING_GROUPS = {

    # originally accommodation split into residential and accommodation
    "residential": [
        "apartments",
        "detached",
        "house",
        "residential",
        "semidetached_house",
        "terrace",
    ],
    
    "accommodation":[
        "barracks",
        "bungalow",
        "cabin",
        "annexe",
        "dormitory",
        "farm",
        "ger",
        "hotel",
        "houseboat",
        "stilt_house",
        "tree_house",
        "trullo",
    ],  
        
    "commercial": [
        "industrial",
        "kiosk",
        "office",
        "retail",
        "supermarket",
        "warehouse",
    ],
    
    "religious": [
        "cathedral",
        "chapel",
        "church",
        "kingdom_hall",
        "monastery",
        "mosque",
        "presbytery",
        "shrine",
        "synagogue",
        "temple",
    ],
    "civic_amenity": [
        "bakehouse",
        "bridge",
        "civic",
        "clock_tower",
        "college",
        "fire_station",
        "government",
        "gatehouse",
        "hospital",
        "kindergarten",
        "museum",
        "public",
        "school",
        "toilets",
        "train_station",
        "transportation",
        "university",
    ],
    "agricultural": [
        "barn",
        "conservatory",
        "cowshed",
        "farm_auxiliary",
        "greenhouse",
        "slurry_tank",
        "stable",
        "sty",
        "livestock",
    ],
    "sports": [
        "grandstand",
        "pavilion",
        "riding_hall",
        "sports_hall",
        "sports_centre",
        "stadium",
    ],
    "storage": [
        "allotment_house",
        "boathouse",
        "hangar",
        "hut",
        "shed",
    ],
    "cars": [
        "carport",
        "garage",
        "garages",
        "parking",
    ],
    "power_technical": [
        "digester",
        "service",
        "tech_cab",
        "transformer_tower",
        "water_tower",
        "storage_tank",
        "silo",
    ],
    "other": [
        "beach_hut",
        "bunker",
        "castle",
        "construction",
        "container",
        "guardhouse",
        "military",
        "outbuilding",
        "pagoda",
        "quonset_hut",
        "roof",
        "ruins",
        "ship",
        "tent",
        "tower",
        "triumphal_arch",
        "windmill",
    ],
}

AMENITY_GROUPS = {
    "sustenance": [
        "bar",
        "biergarten",
        "cafe",
        "fast_food",
        "food_court",
        "ice_cream",
        "pub",
        "restaurant",
    ],
    "education": [
        "college",
        "dancing_school",
        "driving_school",
        "first_aid_school",
        "kindergarten",
        "language_school",
        "library",
        "surf_school",
        "toy_library",
        "research_institute",
        "training",
        "music_school",
        "school",
        "traffic_park",
        "university",
    ],
    "transportation": [
        "bicycle_parking",
        "bicycle_repair_station",
        "bicycle_rental",
        "bicycle_wash",
        "boat_rental",
        "boat_sharing",
        "bus_station",
        "car_rental",
        "car_sharing",
        "car_wash",
        "compressed_air",
        "vehicle_inspection",
        "charging_station",
        "driver_training",
        "ferry_terminal",
        "fuel",
        "grit_bin",
        "motorcycle_parking",
        "parking",
        "parking_entrance",
        "parking_space",
        "taxi",
        "weighbridge",
    ],
    "financial": [
        "atm",
        "payment_terminal",
        "bank",
        "bureau_de_change",
        "money_transfer",
        "payment_centre",
    ],
    "healthcare": [
        "baby_hatch",
        "clinic",
        "dentist",
        "doctors",
        "hospital",
        "nursing_home",
        "pharmacy",
        "social_facility",
        "veterinary",
    ],
    "entertainment_arts_culture": [
        "arts_centre",
        "brothel",
        "casino",
        "cinema",
        "community_centre",
        "conference_centre",
        "events_venue",
        "exhibition_centre",
        "fountain",
        "gambling",
        "love_hotel",
        "music_venue",
        "nightclub",
        "planetarium",
        "public_bookcase",
        "social_centre",
        "stage",
        "stripclub",
        "studio",
        "swingerclub",
        "theatre",
    ],
    "public_service": [
        "courthouse",
        "fire_station",
        "police",
        "post_box",
        "post_depot",
        "post_office",
        "prison",
        "ranger_station",
        "townhall",
    ],
    "facilities": [
        "bbq",
        "bench",
        "check_in",
        "dog_toilet",
        "dressing_room",
        "drinking_water",
        "give_box",
        "lounge",
        "mailroom",
        "parcel_locker",
        "shelter",
        "shower",
        "telephone",
        "toilets",
        "water_point",
        "watering_place",
    ],
    "waste_management": [
        "sanitary_dump_station",
        "recycling",
        "waste_basket",
        "waste_disposal",
        "waste_transfer_station",
    ],
    
    "other": [
        "animal_boarding",
        "animal_breeding",
        "animal_shelter",
        "animal_training",
        "baking_oven",
        "clock",
        "crematorium",
        "dive_centre",
        "grave_yard",
        "hunting_stand",
        "internet_cafe",
        "kitchen",
        "kneipp_water_cure",
        "lounger",
        "marketplace",
        "monastery",
        "mortuary",
        "photo_booth",
        "place_of_mourning",
        "place_of_worship",
        "public_bath",
        "public_building",
        "refugee_site",
    ],
}

PLACE_OF_WORSHIP_GROUPS = {
    "place_of_worship": [
        "chapel",
        "cross",
        "holy_well",
        "husayniyyah",
        "lourdes_grotto",
        "mission_station",
        "mass_rock",
        "monastery",
        "musalla",
        "prayer_flags",
        "shrine",
        "temple",
        "wayside_chapel",
        "wayside_cross",
        "wayside_shrine",
    ]
}

TOURISM_GROUPS = {
    "tourism": [
        "alpine_hut",
        "apartment",
        "aquarium",
        "artwork",
        "attraction",
        "camp_pitch",
        "camp_site",
        "caravan_site",
        "chalet",
        "gallery",
        "guest_house",
        "hostel",
        "hotel",
        "information",
        "motel",
        "museum",
        "picnic_site",
        "theme_park",
        "trail_riding_station",
        "viewpoint",
        "wilderness_hut",
        "zoo",
    ]
}

SHOP_GROUPS = {
    # Food, beverages
    "shop_food_beverages": [
        "alcohol","bakery","beverages","brewing_supplies","butcher","cheese","chocolate","coffee",
        "confectionery","convenience","dairy","deli","farm","food","frozen_food","greengrocer",
        "health_food","ice_cream","nuts","pasta","pastry","seafood","spices","tea","tortilla",
        "water","wine",
    ],

    # General store, department store, mall
    "shop_general": [
        "department_store","general","kiosk","mall","supermarket","wholesale",
    ],

    # Clothing, shoes, accessories
    "shop_clothing_accessories": [
        "baby_goods","bag","boutique","clothes","fabric","fashion","fashion_accessories","jewelry",
        "leather","sewing","shoes","shoe_repair","tailor","watches","wool",
    ],

    # Discount store, charity
    "shop_discount_charity": [
        "charity","second_hand","variety_store",
    ],

    # Health and beauty
    "shop_health_beauty": [
        "beauty","chemist","cosmetics","erotic","hairdresser","hairdresser_supply","hearing_aids",
        "herbalist","massage","medical_supply","nutrition_supplements","optician","perfumery",
        "piercing","tattoo",
    ],

    # Do-it-yourself, household, building materials, gardening
    "shop_diy_garden": [
        "agrarian","appliance","bathroom_furnishing","country_store","doityourself","electrical",
        "energy","fireplace","florist","garden_centre","garden_furniture","gas","glaziery",
        "groundskeeping","hardware","houseware","locksmith","paint","pottery","security",
        "tool_hire","trade",
    ],

    # Furniture and interior
    "shop_furniture_interior": [
        "antiques","bed","candles","carpet","curtain","doors","flooring","furniture",
        "household_linen","interior_decoration","kitchen","lighting","tiles","window_blind",
    ],

    # Electronics
    "shop_electronics": [
        "computer","electronics","hifi","mobile_phone","printer_ink","radiotechnics",
        "telecommunication","vacuum_cleaner",
    ],

    # Outdoors and sport, vehicles
    "shop_outdoors_vehicles": [
        "atv","bicycle","boat","car","car_parts","car_repair","caravan","fishing","fuel","golf",
        "hunting","military_surplus","motorcycle","motorcycle_repair","outdoor","scooter",
        "scuba_diving","ski","snowmobile","sports","surf","swimming_pool","trailer","truck",
        "tyres",
    ],

    # Art, music, hobbies
    "shop_art_hobbies": [
        "art","camera","collector","craft","frame","games","model","music","musical_instrument",
        "photo","trophy","video","video_games",
    ],

    # Stationery, gifts, books, newspapers
    "shop_books_gifts_stationery": [
        "anime","books","gift","lottery","newsagent","stationery","ticket",
    ],

    # Others
    "shop_other": [
        "bookmaker","cannabis","copyshop","dry_cleaning","e-cigarette","funeral_directors",
        "laundry","money_lender","outpost","party","pawnbroker","pest_control","pet",
        "pet_grooming","pyrotechnics","religion","rental","storage_rental","tobacco","toys",
        "travel_agency","vacant","vending_machine","weapons",
    ],
}

OFFICE_GROUPS = {
    # Generic office/service presence
    "office": [
        "company",
        "administrative",
        "administration",
        "central_office",
        "coworking",
        "coworking_space",
        "employment_agency",
        "advertising_agency",
        "telecommunication",
        "it",
        "insurance",
        "real_estate",
        "estate_agent",
        "property_management",
        "lawyer",
        "notary",
        "ngo",
        "association",
        "foundation",
        "political_party",
        "diplomatic",
        "research",
        "educational_institution",
    ],
}

LEISURE_GROUPS = {
    "leisure": [
        "adult_gaming_centre",
        "amusement_arcade",
        "bandstand",
        "bathing_place",
        "beach_resort",
        "bird_hide",
        "bleachers",
        "bowling_alley",
        "common",
        "dance",
        "disc_golf_course",
        "dog_park",
        "escape_game",
        "firepit",
        "fishing",
        "fitness_centre",
        "fitness_station",
        "garden",
        "golf_course",
        "hackerspace",
        "high_ropes_course",
        "horse_riding",
        "ice_rink",
        "marina",
        "miniature_golf",
        "nature_reserve",
        "outdoor_seating",
        "park",
        "picnic_table",
        "pitch",
        "playground",
        "resort",
        "sauna",
        "slipway",
        "sports_centre",
        "sports_hall",
        "stadium",
        "summer_camp",
        "sunbathing",
        "swimming_area",
        "swimming_pool",
        "tanning_salon",
        "track",
        "trampoline_park",
        "water_park",
        "wildlife_hide",
    ]
}

INDUSTRIAL_GROUPS = {
    "industrial": [
        "oil",
        "grinding_mill",
        "factory",
        "wellsite",
        "depot",
        "scrap_yard",
        "gas",              # discouraged on wiki, but you asked to include it
        "warehouse",
        "brickyard",
        "well_cluster",     # deprecated, but you asked to include it
        "port",
        "mine",
        "sawmill",
        "slaughterhouse",
        "cooling",
        "distributor",
        "communication",
        "agriculture",
        "timber",
        "shipyard",
        "heating_station",
        "storage",
        "water",
        "concrete_plant",
        "natural_gas",
        "machine_shop",
        "auto_wrecker",
        "metal_processing",
        "fracking",
        "chemical",
        "electrical",
        "brickworks",
        "refinery",
        "manufacturing",
        "bakery",
        "brewery",
        "logistics",
    ]
}

CRAFT_GROUPS = {
    "craft": [
        "agricultural_engines",
        "atelier",
        "bag_repair",
        "bakery",
        "basket_maker",
        "beekeeper",
        "blacksmith",
        "boatbuilder",
        "bookbinder",
        "brewery",
        "builder",
        "cabinet_maker",
        "candlemaker",
        "car_painter",
        "carpenter",
        "carpet_cleaner",
        "carpet_layer",
        "caterer",
        "chimney_sweeper",
        "cleaning",
        "clockmaker",
        "clothes_mending",
        "confectionery",
        "cooper",
        "dental_technician",
        "distillery",
        "door_construction",
        "dressmaker",
        "electrician",
        "electronics_repair",
        "elevator",
        "embroiderer",
        "engraver",
        "fence_maker",
        "floorer",
        "gardener",
        "glassblower",
        "glaziery",
        "goldsmith",
        "grinding_mill",
        "gunsmith",
        "handicraft",
        "hvac",
        "insulation",
        "interior_decorator",
        "interior_work",
        "jeweller",
        "joiner",
        "key_cutter",
        "laboratory",
        "lapidary",
        "leather",
        "locksmith",
        "luthier",
        "metal_construction",
        "mint",
        "musical_instrument",
        "oil_mill",
        "optician",
        "organ_builder",
        "painter",
        "paperhanger",
        "parquet_layer",
        "paver",
        "pest_control",
        "photographer",
        "photographic_laboratory",
        "photovoltaic",
        "piano_tuner",
        "plasterer",
        "plumber",
        "pottery",
        "printer",
        "printmaker",
        "restoration",
        "rigger",
        "roofer",
        "saddler",
        "sailmaker",
        "sawmill",
        "scaffolder",
        "sculptor",
        "shoemaker",
        "signmaker",
        "stand_builder",
        "stonemason",
        "stove_fitter",
        "sun_protection",
        "tailor",
        "tatami",
        "tiler",
        "tinsmith",
        "toolmaker",
        "turner",
        "upholsterer",
        "watchmaker",
        "water_well_drilling",
        "weaver",
        "welder",
        "window_construction",
        "winery",
    ]
}

# ----------------------------
# Reverse maps: value -> macro-group label
# ----------------------------
def _build_value_to_group(group_name_to_values: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """
    Build value -> group label lookup from dict[group_label -> list(values)].
    Assumes values already normalized.
    """
    out: dict[str, str] = {}
    for group_label, values in group_name_to_values.items():
        for v in values:
            if v:
                out[v] = group_label
    return out

# 1) Single source of truth: domain -> raw GROUPS dict
OSM_DOMAIN_GROUPS = {
    "building": BUILDING_GROUPS,
    "amenity": AMENITY_GROUPS,
    "place_of_worship": PLACE_OF_WORSHIP_GROUPS,
    "tourism": TOURISM_GROUPS,
    "shop": SHOP_GROUPS,
    "leisure": LEISURE_GROUPS,
    "office": OFFICE_GROUPS,
    "craft": CRAFT_GROUPS,
    "industrial": INDUSTRIAL_GROUPS,
    # "landuse": LANDUSE_GROUPS,   # later, one-line add
}

# ----------------------------
# Single source of truth: domain -> (value->group lookup)
# Dict insertion order = precedence.
# ----------------------------
# 2) Derive value->group lookups from the same registry
OSM_DOMAIN_VALUE_TO_GROUP = {
    domain: _build_value_to_group(groups)
    for domain, groups in OSM_DOMAIN_GROUPS.items()
}

# ----------------------------
# Macro-group labels (i.e., the group keys across domains)
# ----------------------------
# Macro group labels = the top-level keys of every *_GROUPS dict
OSM_MACRO_GROUP_LABELS: set[str] = set()
for groups in OSM_DOMAIN_GROUPS.values():
    OSM_MACRO_GROUP_LABELS.update(groups.keys())

# -----------------------------------------------------------------------------
# NORMALIZED RESOLUTION RULES (duplicates + transforms + containers)
# -----------------------------------------------------------------------------
#
# Each BASE token may have:
# - "canonical": force a canonical domain + macro_group
# - "when": conditional variants (based on other-domain presence)
# - "container": parent/child container-drop semantics
#
# Notes:
# - "macro_group" refers to the top-level key inside the *_GROUPS dict
#   (e.g., AMENITY_GROUPS["education"] -> macro_group="education").
# - "domain" refers to OSM domains (building, amenity, shop, ...).
# - Conditions are declarative; implementation interprets them later.
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------
# Place-of-worship suppression: if ANY religious building is present,
# drop ANY place_of_worship-domain token (generic + sub-features).
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Place-of-worship global rule (implemented in apply_resolution_rules)
# ------------------------------------------------------------------
# If ANY religious building exists in the row (domain="building" and base in BUILDING_GROUPS["religious"]),
# then DROP ALL triplets with domain == "place_of_worship" (e.g., mass_rock:place_of_worship:place_of_worship).
POW_RELIGIOUS_BUILDINGS = set(BUILDING_GROUPS["religious"])  # cathedral, church, mosque, ...

RESOLUTION_RULES = {
    # ---------------------------------------------------------
    # Always education (amenity), regardless of source domain/group
    # ---------------------------------------------------------
    "kindergarten": {"canonical": {"domain": "amenity", "macro_group": "education"}},
    "college":      {"canonical": {"domain": "amenity", "macro_group": "education"}},
    "school":       {"canonical": {"domain": "amenity", "macro_group": "education"}},
    "university":   {"canonical": {"domain": "amenity", "macro_group": "education"}},

    # ---------------------------------------------------------
    # Always public_service (amenity), regardless of source domain/group
    # ---------------------------------------------------------
    "fire_station": {"canonical": {"domain": "amenity", "macro_group": "public_service"}},
    "police":       {"canonical": {"domain": "amenity", "macro_group": "public_service"}},

    # ---------------------------------------------------------
    # Always healthcare (amenity), regardless of source domain/group
    # ---------------------------------------------------------
    "hospital": {"canonical": {"domain": "amenity", "macro_group": "healthcare"}},
    "clinic":   {"canonical": {"domain": "amenity", "macro_group": "healthcare"}},

    # ---------------------------------------------------------
    # Always religious (building), regardless of source domain/group
    # ---------------------------------------------------------
    "chapel":    {"canonical": {"domain": "building", "macro_group": "religious"}},
    "monastery": {"canonical": {"domain": "building", "macro_group": "religious"}},
    "temple":    {"canonical": {"domain": "building", "macro_group": "religious"}},

    # ---------------------------------------------------------
    # Always accommodation (building), regardless of source domain/group
    # ---------------------------------------------------------
    "hotel": {"canonical": {"domain": "building", "macro_group": "accommodation"}},
    "house": {"canonical": {"domain": "building", "macro_group": "accommodation"}},

    # ---------------------------------------------------------
    # Museum: canonical to tourism, regardless of source domain/group
    # ---------------------------------------------------------
    "museum": {"canonical": {"domain": "tourism", "macro_group": "tourism"}},

    # ---------------------------------------------------------
    # Pharmacy: canonical to shop taxonomy, regardless of source domain/group
    # ---------------------------------------------------------
    "pharmacy": {"canonical": {"domain": "shop", "macro_group": "shop_health_beauty"}},

    # ---------------------------------------------------------
    # Always shop_food_beverages (shop), regardless of source domain/group
    # ---------------------------------------------------------
    "restaurant": {"canonical": {"domain": "shop", "macro_group": "shop_food_beverages"}},
    "bar":        {"canonical": {"domain": "shop", "macro_group": "shop_food_beverages"}},
    "cafe":       {"canonical": {"domain": "shop", "macro_group": "shop_food_beverages"}},
    "ice_cream":  {"canonical": {"domain": "shop", "macro_group": "shop_food_beverages"}},

    # ---------------------------------------------------------
    # Always entertainment_arts_culture (amenity), regardless of source domain/group
    # ---------------------------------------------------------
    "cinema":           {"canonical": {"domain": "amenity", "macro_group": "entertainment_arts_culture"}},
    "community_centre": {"canonical": {"domain": "amenity", "macro_group": "entertainment_arts_culture"}},

    # ---------------------------------------------------------
    # Always shop categories (shop), regardless of source domain/group
    # ---------------------------------------------------------
    "internet_cafe": {"canonical": {"domain": "shop", "macro_group": "shop_other"}},
    "kiosk":         {"canonical": {"domain": "shop", "macro_group": "shop_general"}},
    "kitchen":       {"canonical": {"domain": "shop", "macro_group": "shop_furniture_interior"}},
    "supermarket":   {"canonical": {"domain": "shop", "macro_group": "shop_general"}},

    # ---------------------------------------------------------
    # Building containers: drop if ANY shop token exists (any base under domain="shop")
    # ---------------------------------------------------------
    "commercial": {
        "container": {
            "parent_domain": "building",
            "drop_parent_if_child": [{"child_domain": "shop", "child_tokens": None}],
        }
    },
    "retail": {
        "container": {
            "parent_domain": "building",
            "drop_parent_if_child": [{"child_domain": "shop", "child_tokens": None}],
        }
    },

    # ---------------------------------------------------------
    # amenity=place_of_worship (generic) -> move into place_of_worship domain taxonomy
    # (then the global POW drop rule may remove it if any religious building exists)
    # ---------------------------------------------------------
    "place_of_worship": {
        "when": [{
            "if": {"domain_equals": "amenity"},
            "then": {"action": "reclassify",
                     "to": {"token": "place_of_worship",
                            "domain": "place_of_worship",
                            "macro_group": "place_of_worship"}},
        }]
    },

    # ---------------------------------------------------------
    # amenity=place_of_mourning -> move into place_of_worship domain taxonomy (keep base token)
    # (then the global POW drop rule may remove it if any religious building exists)
    # ---------------------------------------------------------
    "place_of_mourning": {
        "when": [{
            "if": {"domain_equals": "amenity"},
            "then": {"action": "reclassify",
                     "to": {"token": "place_of_mourning",
                            "domain": "place_of_worship",
                            "macro_group": "place_of_worship"}},
        }]
    },
}

# ----------------------------
# Duplicates (values appearing in multiple GROUPS across domains)
# ----------------------------
def build_duplicate_tokens_map() -> dict[str, set[str]]:
    """
    Values appearing in >1 place across all GROUPS in OSM_DOMAIN_GROUPS.
    origin label format: "<group_label>:<domain>".
    """
    seen: dict[str, set[str]] = defaultdict(set)

    for domain, groups in OSM_DOMAIN_GROUPS.items():
        for group_label, values in groups.items():
            origin = f"{group_label}:{domain}"
            for v in values:
                if v:
                    seen[v].add(origin)

    return {v: origins for v, origins in seen.items() if len(origins) > 1}