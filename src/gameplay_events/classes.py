"""Map CocoDoom / Doom mobj names onto the three detector classes.

Names follow the Chocolate Doom ``mobjtype_t`` enum without the ``MT_``
prefix, which is how CocoDoom stores category names.
"""

ENEMY = "enemy"
HEALTH_ITEM = "health_item"
AMMO_ITEM = "ammo_item"

CLASS_NAMES = (ENEMY, HEALTH_ITEM, AMMO_ITEM)

# Living monsters. Corpses keep these names in CocoDoom; the dataset
# builder drops flat boxes so a dead monster is not trained as an enemy.
ENEMY_TYPES = frozenset(
    {
        "POSSESSED",
        "SHOTGUY",
        "VILE",
        "UNDEAD",
        "FATSO",
        "CHAINGUY",
        "TROOP",
        "SERGEANT",
        "SHADOWS",
        "HEAD",
        "BRUISER",
        "KNIGHT",
        "SKULL",
        "SPIDER",
        "BABY",
        "CYBORG",
        "PAIN",
        "WOLFSS",
        "KEEN",
        "BOSSBRAIN",
    }
)

# Stimpack, medikit, health bonus, soulsphere, megasphere.
HEALTH_TYPES = frozenset({"MISC2", "MISC10", "MISC11", "MISC12", "MEGA"})

# Bullets, shells, rockets, cells, and their boxes.
AMMO_TYPES = frozenset(
    {
        "CLIP",
        "MISC17",
        "MISC18",
        "MISC19",
        "MISC20",
        "MISC21",
        "MISC22",
        "MISC23",
    }
)

CLASS_OF = {
    **{name: ENEMY for name in ENEMY_TYPES},
    **{name: HEALTH_ITEM for name in HEALTH_TYPES},
    **{name: AMMO_ITEM for name in AMMO_TYPES},
}

# A living sprite is taller than a corpse. Boxes flatter than this are
# left out of the enemy class so a kill shows up as a disappearance.
MIN_ENEMY_ASPECT = 0.6


def class_index(class_name):
    return CLASS_NAMES.index(class_name)


def split_for_map(map_id):
    """Published CocoDoom map split.

    Train maps are ``(id - 1) % 4`` in {0, 1}, validation is 2, test is 3.
    """
    group = (map_id - 1) % 4
    if group <= 1:
        return "train"
    if group == 2:
        return "val"
    return "test"
