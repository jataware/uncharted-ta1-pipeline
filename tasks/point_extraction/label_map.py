from enum import Enum


# ontology for common / high priority YOLO point classes
class POINT_CLASS(str, Enum):
    STRIKE_AND_DIP = "strike_and_dip"  # aka inclined bedding
    HORIZONTAL_BEDDING = "horizontal_bedding"
    OVERTURNED_BEDDING = "overturned_bedding"
    VERTICAL_BEDDING = "vertical_bedding"
    INCLINED_FOLIATION = "inclined_foliation"  # line with solid triangle
    INCLINED_FOLIATION_IGNEOUS = "inclined_foliation_igneous"  # with hollow triangle
    VERTICAL_FOLIATION = "vertical_foliation"
    VERTICAL_JOINT = "vertical_joint"
    SINK_HOLE = "sink_hole"
    LINEATION = "lineation"
    DRILL_HOLE = "drill_hole"
    GRAVEL_BORROW_PIT = "gravel_borrow_pit"
    MINE_SHAFT = "mine_shaft"
    PROSPECT = "prospect"
    MINE_TUNNEL = "mine_tunnel"  # aka adit or "4_pt"
    MINE_QUARRY = "mine_quarry"

    def __str__(self):
        return self.value


# mapping of YOLO model classes to output CDR ontology for common point symbols
# (if different than LARA's internal ontology)
YOLO_TO_CDR_LABEL = {
    POINT_CLASS.STRIKE_AND_DIP: "inclined_bedding",
    POINT_CLASS.INCLINED_FOLIATION: "inclined_foliation_metamorphic",
    POINT_CLASS.INCLINED_FOLIATION_IGNEOUS: "inclined_foliation_igneous",
    POINT_CLASS.GRAVEL_BORROW_PIT: "pit",
    POINT_CLASS.MINE_QUARRY: "quarry",
}


# mapping of YOLO model classes to legend item labels
LABEL_MAPPING = {
    # --- STRIKE and DIP (aka INCLINED BEDDING)
    "strike_and_dip": [
        "strike_and_dip",
        "strike_dip_beds",
        "bedding_inclined",
        "inclined_bedding",
        "bedding_pt",
        "inclined_pt",
        "strikedip",
    ],
    # --- HORIZONTAL BEDDING
    "horizontal_bedding": [
        "horizontal_bedding",
        "bedding_horizontal",
        "horiz_bedding",
        "bedding_horiz",
        "horizbed",
        "horizon_bedding",
    ],
    # --- OVERTURNED BEDDING
    "overturned_bedding": [
        "overturned_bedding",
        "bedding_overturned",
        "overturn_bedding",
    ],
    # --- VERTICAL BEDDING
    "vertical_bedding": [
        "vertical_bedding",
        "bedding_vertical",
    ],
    # --- INCLINED FOLIATION (with solid triangle)
    "inclined_foliation": [
        "inclined_foliation",
        "foliation_inclined",
        "incl_meta_foliation_pt",
    ],
    # --- INCLINED FOLIATION IGNEOUS (with hollow triangle)
    "inclined_foliation_igneous": [
        "inclined_foliation_igneous",
        "igneous_foliation",
        "platy_parting",
    ],
    # --- VERTICAL FOLIATION
    "vertical_foliation": [
        "vertical_foliation",
        "vert_meta_foliation",
        "GrandCanyon_vertical_joint_pt",
    ],
    # --- VERTICAL JOINT (line with square in middle, solid or hollow)
    "vertical_joint": ["vertical_joint", "joint_vertical"],
    # --- GRAVEL PIT or BORROW PIT  (1_pt -- two crossed shovels)
    "gravel_borrow_pit": [
        "gravel_borrow_pit",
        "geo_mosaic_1_pt",
    ],
    # --- MINE SHAFT    (2_pt -- square, black and white)
    "mine_shaft": [
        "mine_shaft",
        "geo_mosaic_2_pt",
    ],
    # --- PROSPECT PIT  (3_pt -- tall X)
    "prospect": [
        "prospect",
        "prospect_pit",
        "geo_mosaic_3_pt",
    ],
    # --- MINE TUNNEL (4_pt -- aka ADIT, rotated Y)
    "mine_tunnel": ["mine_tunnel", "geo_mosaic_4_pt", "adit"],
    # --- MINE or QUARRY or OPEN PIT    (5_pt -- crossed pick-axes)
    "mine_quarry": ["mine_quarry", "geo_mosaic_5_pt"],
    # --- SINK HOLE
    "sink_hole": ["sink_hole", "sinkhole_pt"],
    # --- LINEATION
    "lineation": ["lineation", "lineation_pt"],
    # --- DRILL HOLE
    "drill_hole": ["drill_hole", "drillhole_pt"],
}
