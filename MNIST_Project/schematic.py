import mcschematic
from barrel import block

schem = mcschematic.MCSchematic()



schem.setBlock(
    (0, 0, 0),
    '''
    minecraft:barrel[facing=up]{
        Items:[
            {Slot:0b,id:"minecraft:totem_of_undying",Count:1b},
            {Slot:1b,id:"minecraft:emerald",Count:32b},
            {Slot:2b,id:"minecraft:netherite_ingot",Count:16b}
        ]
    }
    '''.replace("\n", "").replace(" ", "")
)


# schem.save(
#     outputFolderPath="output",
#     schemName="tower",
#     version=mcschematic.Version.JE_26_1_2
# )
