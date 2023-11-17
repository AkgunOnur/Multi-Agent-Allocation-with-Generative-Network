from monobeast_amigo import *

def main(flags):
    map_file = "doorkey_all_maps.pickle"
    target_index = 2
    flags.savedir = "Nov17_MiniGrid/DoorKey/map" + str(target_index)

    with open(map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)

    target_map = new_target_maps[target_index]

    train(flags, target_map=target_map)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)