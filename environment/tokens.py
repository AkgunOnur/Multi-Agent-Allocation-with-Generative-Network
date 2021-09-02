from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

GROUND_TOKENS = OrderedDict(
    {
        "-": "Empty Ground"
    }
)


OBSTACLE_TOKENS = OrderedDict(
    {
        "W": "Obstacle Block"
    }
)


PRIZE_TOKENS = OrderedDict(
    {
        "X": "Prize"
    }
)


SPECIAL_TOKENS = OrderedDict(
    {
        "D": "Drone Starting Position, not having it will force the engine to start at x = 0 and the first ground floor."
    }
)


TOKEN_DOWNSAMPLING_HIERARCHY = [
    GROUND_TOKENS,
    OBSTACLE_TOKENS,
    PRIZE_TOKENS,
    SPECIAL_TOKENS
]


TOKENS = OrderedDict(
    {**GROUND_TOKENS, **OBSTACLE_TOKENS, **PRIZE_TOKENS, **SPECIAL_TOKENS}
)

TOKEN_GROUPS = [GROUND_TOKENS, OBSTACLE_TOKENS, PRIZE_TOKENS, SPECIAL_TOKENS]

REPLACE_TOKENS = {"D": "-"}  # We replace these tokens so the generator doesn't add random start or end points

