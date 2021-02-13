USED_NAMES = dict()
USED_IDS = set()
MAX_ID = 1000000


def get_default_name(obj):
    global USED_NAMES
    name_id = USED_NAMES.get(obj.__class__.__name__)
    name_id = 0 if name_id is None else name_id + 1
    USED_NAMES[obj.__class__.__name__] = name_id

    name = obj.__class__.__name__ if name_id == 0 else \
           obj.__class__.__name__ + "_{}".format(name_id)

    return name


def reset_used_names():
    global USED_NAMES
    USED_NAMES = dict()


def get_available_id():
    global USED_IDS
    global MAX_ID

    if len(USED_IDS) == 0:
        new_id = 0
        USED_IDS.add(new_id)
        return new_id
    else:
        biggest_id = max(USED_IDS)
        if biggest_id >= MAX_ID:
            raise ValueError("Cannot assign more ID!")
        else:
            new_id = biggest_id + 1
            USED_IDS.add(new_id)
            return new_id


def reset_used_ids():
    global USED_IDS
    USED_IDS = set()
