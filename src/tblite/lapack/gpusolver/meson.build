srcs += files(
    'sygvd.F90',
)
if has_gpu
  srcs += files('cuda.c')
endif
if get_option('cusolvermp')
  srcs += files('cusolvermp.c')
endif
