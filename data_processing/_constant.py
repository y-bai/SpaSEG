"""Constants that user will deal with"""

class PositionScale():
    # scale the positional value of array_row, array_col from adata.obsm["spatial"]
    MERFISH_SCALE_FACTOR = 500
    SEQFISH_SCALE_FACTOR = 400
    SLIDESEQV2_SCALE_FACTOR = 500

class SpotSize():
    MERFISH_SPOT_SIZE = 20
    SEQFISH_SPOT_SIZE = 0.03
    SLIDESEQV2_SPOT_SIZE = 15
    VISIUM_SPOT_SIZE = 100
    STEREO_SPOT_SIZE = 1


class CellbinSize():
    # The average size of mammalian cell is approximately equal to Stereo-seq bin14*bin14
    CELLBIN_SIZE = 14