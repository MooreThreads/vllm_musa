#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, mt_bfloat16, mt_bfloat16, mt_bfloat16)
FOR_INST_BGMV_WIDE_NARROW(INST_BGMV_ONESIDE, mt_bfloat16, mt_bfloat16, mt_bfloat16)
