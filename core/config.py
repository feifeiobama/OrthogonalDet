def add_config(cfg):
    cfg.MODEL.NUM_CLASSES = 81
    cfg.MODEL.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.NHEADS = 8
    cfg.MODEL.DROPOUT = 0.0
    cfg.MODEL.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ACTIVATION = 'relu'
    cfg.MODEL.HIDDEN_DIM = 256
    cfg.MODEL.NUM_CLS = 1
    cfg.MODEL.NUM_REG = 3
    cfg.MODEL.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.NUM_DYNAMIC = 2
    cfg.MODEL.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.CLASS_WEIGHT = 2.0
    cfg.MODEL.NC = True
    cfg.MODEL.NC_WEIGHT = 0.1
    cfg.MODEL.GIOU_WEIGHT = 2.0
    cfg.MODEL.L1_WEIGHT = 5.0
    cfg.MODEL.DEEP_SUPERVISION = True
    cfg.MODEL.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.ALPHA = 0.25
    cfg.MODEL.GAMMA = 2.0
    cfg.MODEL.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.OTA_K = 5
    cfg.MODEL.FORWARD_K = 10

    # WARM_UP
    cfg.MODEL.CHANGE_START = 0

    # Diffusion
    cfg.MODEL.SNR_SCALE = 2.0
    cfg.MODEL.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.USE_NMS = True
    cfg.MODEL.M_STEP = 20
    cfg.MODEL.SAMPLING_METHOD = 'Random_'

    # Disentanglement
    cfg.MODEL.DISENTANGLED = 2  # 0: RandBox, 1: separate head, 2: feature orthogonality
    cfg.MODEL.DECORR_WEIGHT = 1.  # weight for prediction decorrelation loss

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # OW EVALUATION
    cfg.TEST.PREV_INTRODUCED_CLS = 0
    cfg.TEST.CUR_INTRODUCED_CLS = 20
    cfg.TEST.PREV_CLASSES = ()  # previously seen classes
    cfg.TEST.MASK = 1  # 0: no mask, 1: mask unseen classes, 2: mask prev and unseen classes
    cfg.TEST.SCORE_THRESH = 0.15  # follow RandBox
