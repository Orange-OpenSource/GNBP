ހ
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ǈ

NoOpNoOp
?	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?	
value?	B?	 B?	
}
	lbcpe

modulation
	variables
regularization_losses
trainable_variables
	keras_api

signatures
_
product
		variables

regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
?
	variables
regularization_losses
metrics
layer_metrics

layers
layer_regularization_losses
non_trainable_variables
trainable_variables
 
R
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
?
		variables

regularization_losses
layer_metrics

layers
layer_regularization_losses
metrics
non_trainable_variables
trainable_variables
 
 
 
?
	variables
regularization_losses
layer_metrics

 layers
!layer_regularization_losses
"metrics
#non_trainable_variables
trainable_variables
 
 

0
1
 
 
 
 
 
?
	variables
regularization_losses
$layer_metrics

%layers
&layer_regularization_losses
'metrics
(non_trainable_variables
trainable_variables
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
PartitionedCallPartitionedCallserving_default_input_1serving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_555360
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_555722
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_555732??
?
l
__inference__traced_save_555722
file_prefix
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
?
d
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555673

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_5552352
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
(__inference_encoder_layer_call_fn_555542
input_1
input_2
identity?
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_5552382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?1
?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555310

inputs
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y[
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
sub?
&product_with_external_weights_29/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_29/Shape?
4product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????26
4product_with_external_weights_29/strided_slice/stack?
6product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_29/strided_slice/stack_1?
6product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_29/strided_slice/stack_2?
.product_with_external_weights_29/strided_sliceStridedSlice/product_with_external_weights_29/Shape:output:0=product_with_external_weights_29/strided_slice/stack:output:0?product_with_external_weights_29/strided_slice/stack_1:output:0?product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_29/strided_slice?
0product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0product_with_external_weights_29/Reshape/shape/1?
.product_with_external_weights_29/Reshape/shapePack7product_with_external_weights_29/strided_slice:output:09product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_29/Reshape/shape?
(product_with_external_weights_29/ReshapeReshapeinputs_17product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2*
(product_with_external_weights_29/Reshape?
/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_29/ExpandDims/dim?
+product_with_external_weights_29/ExpandDims
ExpandDimssub:z:08product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2-
+product_with_external_weights_29/ExpandDims?
&product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_29/stack?
%product_with_external_weights_29/TileTile4product_with_external_weights_29/ExpandDims:output:0/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2'
%product_with_external_weights_29/Tile?
/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_29/transpose/perm?
*product_with_external_weights_29/transpose	Transpose1product_with_external_weights_29/Reshape:output:08product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2,
*product_with_external_weights_29/transpose?
0product_with_external_weights_29/PartitionedCallPartitionedCall.product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_55514722
0product_with_external_weights_29/PartitionedCall?
$product_with_external_weights_29/MulMul.product_with_external_weights_29/Tile:output:09product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/Mul?
1product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_29/transpose_1/perm?
,product_with_external_weights_29/transpose_1	Transpose1product_with_external_weights_29/Reshape:output:0:product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2.
,product_with_external_weights_29/transpose_1?
2product_with_external_weights_29/PartitionedCall_1PartitionedCall0product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_55515724
2product_with_external_weights_29/PartitionedCall_1?
$product_with_external_weights_29/addAddV2(product_with_external_weights_29/Mul:z:0;product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/add?
7product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7product_with_external_weights_29/Prod/reduction_indices?
%product_with_external_weights_29/ProdProd(product_with_external_weights_29/add:z:0@product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2'
%product_with_external_weights_29/Prods
NegNeg.product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
o
C__inference_encoder_layer_call_and_return_conditional_losses_555404
inputs_0
inputs_1
identity?
:linear_block_code_product_encoder_with_external_g_29/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_29/mul/y?
8linear_block_code_product_encoder_with_external_g_29/mulMulinputs_0Clinear_block_code_product_encoder_with_external_g_29/mul/y:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/mul?
:linear_block_code_product_encoder_with_external_g_29/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:linear_block_code_product_encoder_with_external_g_29/sub/x?
8linear_block_code_product_encoder_with_external_g_29/subSubClinear_block_code_product_encoder_with_external_g_29/sub/x:output:0<linear_block_code_product_encoder_with_external_g_29/mul:z:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/sub?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ShapeShape<linear_block_code_product_encoder_with_external_g_29/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape?
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2k
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape:output:0rlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape?
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ReshapeReshapeinputs_1llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2_
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim?
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_29/sub:z:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2b
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/TileTileilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm?
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2a
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_5551472g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/MulMulclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul?
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm?
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0olinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2c
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1?
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_5551572i
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/addAddV2]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul:z:0plinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add?
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2n
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ProdProd]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add:z:0ulinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod?
8linear_block_code_product_encoder_with_external_g_29/NegNegclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/Neg?
<linear_block_code_product_encoder_with_external_g_29/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_29/Neg:y:0*
T0*'
_output_shapes
:?????????2>
<linear_block_code_product_encoder_with_external_g_29/Sigmoid?
-differentiable_bpsk_modulation_layer_29/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_29/sub/y?
+differentiable_bpsk_modulation_layer_29/subSub@linear_block_code_product_encoder_with_external_g_29/Sigmoid:y:06differentiable_bpsk_modulation_layer_29/sub/y:output:0*
T0*'
_output_shapes
:?????????2-
+differentiable_bpsk_modulation_layer_29/sub?
1differentiable_bpsk_modulation_layer_29/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_29/Greater/y?
/differentiable_bpsk_modulation_layer_29/GreaterGreater/differentiable_bpsk_modulation_layer_29/sub:z:0:differentiable_bpsk_modulation_layer_29/Greater/y:output:0*
T0*'
_output_shapes
:?????????21
/differentiable_bpsk_modulation_layer_29/Greater?
2differentiable_bpsk_modulation_layer_29/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/t?
2differentiable_bpsk_modulation_layer_29/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/e?
0differentiable_bpsk_modulation_layer_29/SelectV2SelectV23differentiable_bpsk_modulation_layer_29/Greater:z:0;differentiable_bpsk_modulation_layer_29/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_29/SelectV2/e:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/SelectV2?
0differentiable_bpsk_modulation_layer_29/IdentityIdentity9differentiable_bpsk_modulation_layer_29/SelectV2:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/Identity?
1differentiable_bpsk_modulation_layer_29/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_29/SelectV2:output:0/differentiable_bpsk_modulation_layer_29/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555394*:
_output_shapes(
&:?????????:?????????23
1differentiable_bpsk_modulation_layer_29/IdentityN?
IdentityIdentity:differentiable_bpsk_modulation_layer_29/IdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?l
K
!__inference__wrapped_model_555176
input_1
input_2
identity?
Bencoder/linear_block_code_product_encoder_with_external_g_29/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2D
Bencoder/linear_block_code_product_encoder_with_external_g_29/mul/y?
@encoder/linear_block_code_product_encoder_with_external_g_29/mulMulinput_1Kencoder/linear_block_code_product_encoder_with_external_g_29/mul/y:output:0*
T0*'
_output_shapes
:?????????2B
@encoder/linear_block_code_product_encoder_with_external_g_29/mul?
Bencoder/linear_block_code_product_encoder_with_external_g_29/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
Bencoder/linear_block_code_product_encoder_with_external_g_29/sub/x?
@encoder/linear_block_code_product_encoder_with_external_g_29/subSubKencoder/linear_block_code_product_encoder_with_external_g_29/sub/x:output:0Dencoder/linear_block_code_product_encoder_with_external_g_29/mul:z:0*
T0*'
_output_shapes
:?????????2B
@encoder/linear_block_code_product_encoder_with_external_g_29/sub?
cencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ShapeShapeDencoder/linear_block_code_product_encoder_with_external_g_29/sub:z:0*
T0*
_output_shapes
:2e
cencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape?
qencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2s
qencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack?
sencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2u
sencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1?
sencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2u
sencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2?
kencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_sliceStridedSlicelencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape:output:0zencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack:output:0|encoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1:output:0|encoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2m
kencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice?
mencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2o
mencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1?
kencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shapePacktencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice:output:0vencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2m
kencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape?
eencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ReshapeReshapeinput_2tencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2g
eencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape?
lencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2n
lencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim?
hencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims
ExpandDimsDencoder/linear_block_code_product_encoder_with_external_g_29/sub:z:0uencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2j
hencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims?
cencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2e
cencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack?
bencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/TileTileqencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims:output:0lencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2d
bencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile?
lencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2n
lencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm?
gencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose	Transposenencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0uencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2i
gencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose?
mencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCallPartitionedCallkencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_5551472o
mencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall?
aencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/MulMulkencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile:output:0vencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2c
aencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul?
nencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2p
nencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm?
iencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1	Transposenencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0wencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2k
iencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1?
oencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1PartitionedCallmencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_5551572q
oencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1?
aencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/addAddV2eencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul:z:0xencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2c
aencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add?
tencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2v
tencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices?
bencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ProdProdeencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add:z:0}encoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2d
bencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod?
@encoder/linear_block_code_product_encoder_with_external_g_29/NegNegkencoder/linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2B
@encoder/linear_block_code_product_encoder_with_external_g_29/Neg?
Dencoder/linear_block_code_product_encoder_with_external_g_29/SigmoidSigmoidDencoder/linear_block_code_product_encoder_with_external_g_29/Neg:y:0*
T0*'
_output_shapes
:?????????2F
Dencoder/linear_block_code_product_encoder_with_external_g_29/Sigmoid?
5encoder/differentiable_bpsk_modulation_layer_29/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?27
5encoder/differentiable_bpsk_modulation_layer_29/sub/y?
3encoder/differentiable_bpsk_modulation_layer_29/subSubHencoder/linear_block_code_product_encoder_with_external_g_29/Sigmoid:y:0>encoder/differentiable_bpsk_modulation_layer_29/sub/y:output:0*
T0*'
_output_shapes
:?????????25
3encoder/differentiable_bpsk_modulation_layer_29/sub?
9encoder/differentiable_bpsk_modulation_layer_29/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9encoder/differentiable_bpsk_modulation_layer_29/Greater/y?
7encoder/differentiable_bpsk_modulation_layer_29/GreaterGreater7encoder/differentiable_bpsk_modulation_layer_29/sub:z:0Bencoder/differentiable_bpsk_modulation_layer_29/Greater/y:output:0*
T0*'
_output_shapes
:?????????29
7encoder/differentiable_bpsk_modulation_layer_29/Greater?
:encoder/differentiable_bpsk_modulation_layer_29/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:encoder/differentiable_bpsk_modulation_layer_29/SelectV2/t?
:encoder/differentiable_bpsk_modulation_layer_29/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:encoder/differentiable_bpsk_modulation_layer_29/SelectV2/e?
8encoder/differentiable_bpsk_modulation_layer_29/SelectV2SelectV2;encoder/differentiable_bpsk_modulation_layer_29/Greater:z:0Cencoder/differentiable_bpsk_modulation_layer_29/SelectV2/t:output:0Cencoder/differentiable_bpsk_modulation_layer_29/SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2:
8encoder/differentiable_bpsk_modulation_layer_29/SelectV2?
8encoder/differentiable_bpsk_modulation_layer_29/IdentityIdentityAencoder/differentiable_bpsk_modulation_layer_29/SelectV2:output:0*
T0*'
_output_shapes
:?????????2:
8encoder/differentiable_bpsk_modulation_layer_29/Identity?
9encoder/differentiable_bpsk_modulation_layer_29/IdentityN	IdentityNAencoder/differentiable_bpsk_modulation_layer_29/SelectV2:output:07encoder/differentiable_bpsk_modulation_layer_29/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555166*:
_output_shapes(
&:?????????:?????????2;
9encoder/differentiable_bpsk_modulation_layer_29/IdentityN?
IdentityIdentityBencoder/differentiable_bpsk_modulation_layer_29/IdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
m
C__inference_encoder_layer_call_and_return_conditional_losses_555238

inputs
inputs_1
identity?
Dlinear_block_code_product_encoder_with_external_g_29/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_5552182F
Dlinear_block_code_product_encoder_with_external_g_29/PartitionedCall?
7differentiable_bpsk_modulation_layer_29/PartitionedCallPartitionedCallMlinear_block_code_product_encoder_with_external_g_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_55523529
7differentiable_bpsk_modulation_layer_29/PartitionedCall?
IdentityIdentity@differentiable_bpsk_modulation_layer_29/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
m
C__inference_encoder_layer_call_and_return_conditional_losses_555492
input_1
input_2
identity?
:linear_block_code_product_encoder_with_external_g_29/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_29/mul/y?
8linear_block_code_product_encoder_with_external_g_29/mulMulinput_1Clinear_block_code_product_encoder_with_external_g_29/mul/y:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/mul?
:linear_block_code_product_encoder_with_external_g_29/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:linear_block_code_product_encoder_with_external_g_29/sub/x?
8linear_block_code_product_encoder_with_external_g_29/subSubClinear_block_code_product_encoder_with_external_g_29/sub/x:output:0<linear_block_code_product_encoder_with_external_g_29/mul:z:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/sub?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ShapeShape<linear_block_code_product_encoder_with_external_g_29/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape?
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2k
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape:output:0rlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape?
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ReshapeReshapeinput_2llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2_
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim?
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_29/sub:z:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2b
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/TileTileilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm?
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2a
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_5551472g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/MulMulclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul?
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm?
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0olinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2c
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1?
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_5551572i
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/addAddV2]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul:z:0plinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add?
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2n
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ProdProd]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add:z:0ulinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod?
8linear_block_code_product_encoder_with_external_g_29/NegNegclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/Neg?
<linear_block_code_product_encoder_with_external_g_29/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_29/Neg:y:0*
T0*'
_output_shapes
:?????????2>
<linear_block_code_product_encoder_with_external_g_29/Sigmoid?
-differentiable_bpsk_modulation_layer_29/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_29/sub/y?
+differentiable_bpsk_modulation_layer_29/subSub@linear_block_code_product_encoder_with_external_g_29/Sigmoid:y:06differentiable_bpsk_modulation_layer_29/sub/y:output:0*
T0*'
_output_shapes
:?????????2-
+differentiable_bpsk_modulation_layer_29/sub?
1differentiable_bpsk_modulation_layer_29/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_29/Greater/y?
/differentiable_bpsk_modulation_layer_29/GreaterGreater/differentiable_bpsk_modulation_layer_29/sub:z:0:differentiable_bpsk_modulation_layer_29/Greater/y:output:0*
T0*'
_output_shapes
:?????????21
/differentiable_bpsk_modulation_layer_29/Greater?
2differentiable_bpsk_modulation_layer_29/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/t?
2differentiable_bpsk_modulation_layer_29/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/e?
0differentiable_bpsk_modulation_layer_29/SelectV2SelectV23differentiable_bpsk_modulation_layer_29/Greater:z:0;differentiable_bpsk_modulation_layer_29/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_29/SelectV2/e:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/SelectV2?
0differentiable_bpsk_modulation_layer_29/IdentityIdentity9differentiable_bpsk_modulation_layer_29/SelectV2:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/Identity?
1differentiable_bpsk_modulation_layer_29/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_29/SelectV2:output:0/differentiable_bpsk_modulation_layer_29/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555482*:
_output_shapes(
&:?????????:?????????23
1differentiable_bpsk_modulation_layer_29/IdentityN?
IdentityIdentity:differentiable_bpsk_modulation_layer_29/IdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
N
$__inference_signature_wrapper_555360
input_1
input_2
identity?
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_5551762
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?e
m
C__inference_encoder_layer_call_and_return_conditional_losses_555536
input_1
input_2
identity?
:linear_block_code_product_encoder_with_external_g_29/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_29/mul/y?
8linear_block_code_product_encoder_with_external_g_29/mulMulinput_1Clinear_block_code_product_encoder_with_external_g_29/mul/y:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/mul?
:linear_block_code_product_encoder_with_external_g_29/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:linear_block_code_product_encoder_with_external_g_29/sub/x?
8linear_block_code_product_encoder_with_external_g_29/subSubClinear_block_code_product_encoder_with_external_g_29/sub/x:output:0<linear_block_code_product_encoder_with_external_g_29/mul:z:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/sub?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ShapeShape<linear_block_code_product_encoder_with_external_g_29/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape?
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2k
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape:output:0rlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape?
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ReshapeReshapeinput_2llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2_
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim?
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_29/sub:z:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2b
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/TileTileilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm?
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2a
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_5551472g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/MulMulclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul?
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm?
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0olinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2c
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1?
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_5551572i
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/addAddV2]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul:z:0plinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add?
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2n
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ProdProd]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add:z:0ulinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod?
8linear_block_code_product_encoder_with_external_g_29/NegNegclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/Neg?
<linear_block_code_product_encoder_with_external_g_29/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_29/Neg:y:0*
T0*'
_output_shapes
:?????????2>
<linear_block_code_product_encoder_with_external_g_29/Sigmoid?
-differentiable_bpsk_modulation_layer_29/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_29/sub/y?
+differentiable_bpsk_modulation_layer_29/subSub@linear_block_code_product_encoder_with_external_g_29/Sigmoid:y:06differentiable_bpsk_modulation_layer_29/sub/y:output:0*
T0*'
_output_shapes
:?????????2-
+differentiable_bpsk_modulation_layer_29/sub?
1differentiable_bpsk_modulation_layer_29/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_29/Greater/y?
/differentiable_bpsk_modulation_layer_29/GreaterGreater/differentiable_bpsk_modulation_layer_29/sub:z:0:differentiable_bpsk_modulation_layer_29/Greater/y:output:0*
T0*'
_output_shapes
:?????????21
/differentiable_bpsk_modulation_layer_29/Greater?
2differentiable_bpsk_modulation_layer_29/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/t?
2differentiable_bpsk_modulation_layer_29/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/e?
0differentiable_bpsk_modulation_layer_29/SelectV2SelectV23differentiable_bpsk_modulation_layer_29/Greater:z:0;differentiable_bpsk_modulation_layer_29/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_29/SelectV2/e:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/SelectV2?
0differentiable_bpsk_modulation_layer_29/IdentityIdentity9differentiable_bpsk_modulation_layer_29/SelectV2:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/Identity?
1differentiable_bpsk_modulation_layer_29/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_29/SelectV2:output:0/differentiable_bpsk_modulation_layer_29/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555526*:
_output_shapes(
&:?????????:?????????23
1differentiable_bpsk_modulation_layer_29/IdentityN?
IdentityIdentity:differentiable_bpsk_modulation_layer_29/IdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
+
__inference_f_555682
w
identityL
IdentityIdentityw*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::A =

_output_shapes

:

_user_specified_namew
?1
?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555593
inputs_0
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y]
mulMulinputs_0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
sub?
&product_with_external_weights_29/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_29/Shape?
4product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????26
4product_with_external_weights_29/strided_slice/stack?
6product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_29/strided_slice/stack_1?
6product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_29/strided_slice/stack_2?
.product_with_external_weights_29/strided_sliceStridedSlice/product_with_external_weights_29/Shape:output:0=product_with_external_weights_29/strided_slice/stack:output:0?product_with_external_weights_29/strided_slice/stack_1:output:0?product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_29/strided_slice?
0product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0product_with_external_weights_29/Reshape/shape/1?
.product_with_external_weights_29/Reshape/shapePack7product_with_external_weights_29/strided_slice:output:09product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_29/Reshape/shape?
(product_with_external_weights_29/ReshapeReshapeinputs_17product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2*
(product_with_external_weights_29/Reshape?
/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_29/ExpandDims/dim?
+product_with_external_weights_29/ExpandDims
ExpandDimssub:z:08product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2-
+product_with_external_weights_29/ExpandDims?
&product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_29/stack?
%product_with_external_weights_29/TileTile4product_with_external_weights_29/ExpandDims:output:0/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2'
%product_with_external_weights_29/Tile?
/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_29/transpose/perm?
*product_with_external_weights_29/transpose	Transpose1product_with_external_weights_29/Reshape:output:08product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2,
*product_with_external_weights_29/transpose?
0product_with_external_weights_29/PartitionedCallPartitionedCall.product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_55514722
0product_with_external_weights_29/PartitionedCall?
$product_with_external_weights_29/MulMul.product_with_external_weights_29/Tile:output:09product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/Mul?
1product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_29/transpose_1/perm?
,product_with_external_weights_29/transpose_1	Transpose1product_with_external_weights_29/Reshape:output:0:product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2.
,product_with_external_weights_29/transpose_1?
2product_with_external_weights_29/PartitionedCall_1PartitionedCall0product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_55515724
2product_with_external_weights_29/PartitionedCall_1?
$product_with_external_weights_29/addAddV2(product_with_external_weights_29/Mul:z:0;product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/add?
7product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7product_with_external_weights_29/Prod/reduction_indices?
%product_with_external_weights_29/ProdProd(product_with_external_weights_29/add:z:0@product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2'
%product_with_external_weights_29/Prods
NegNeg.product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
+
__inference_f_555686
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namew
?
+
__inference_g_555157
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namew
?1
?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555218

inputs
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y[
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
sub?
&product_with_external_weights_29/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_29/Shape?
4product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????26
4product_with_external_weights_29/strided_slice/stack?
6product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_29/strided_slice/stack_1?
6product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_29/strided_slice/stack_2?
.product_with_external_weights_29/strided_sliceStridedSlice/product_with_external_weights_29/Shape:output:0=product_with_external_weights_29/strided_slice/stack:output:0?product_with_external_weights_29/strided_slice/stack_1:output:0?product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_29/strided_slice?
0product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0product_with_external_weights_29/Reshape/shape/1?
.product_with_external_weights_29/Reshape/shapePack7product_with_external_weights_29/strided_slice:output:09product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_29/Reshape/shape?
(product_with_external_weights_29/ReshapeReshapeinputs_17product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2*
(product_with_external_weights_29/Reshape?
/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_29/ExpandDims/dim?
+product_with_external_weights_29/ExpandDims
ExpandDimssub:z:08product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2-
+product_with_external_weights_29/ExpandDims?
&product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_29/stack?
%product_with_external_weights_29/TileTile4product_with_external_weights_29/ExpandDims:output:0/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2'
%product_with_external_weights_29/Tile?
/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_29/transpose/perm?
*product_with_external_weights_29/transpose	Transpose1product_with_external_weights_29/Reshape:output:08product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2,
*product_with_external_weights_29/transpose?
0product_with_external_weights_29/PartitionedCallPartitionedCall.product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_55514722
0product_with_external_weights_29/PartitionedCall?
$product_with_external_weights_29/MulMul.product_with_external_weights_29/Tile:output:09product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/Mul?
1product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_29/transpose_1/perm?
,product_with_external_weights_29/transpose_1	Transpose1product_with_external_weights_29/Reshape:output:0:product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2.
,product_with_external_weights_29/transpose_1?
2product_with_external_weights_29/PartitionedCall_1PartitionedCall0product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_55515724
2product_with_external_weights_29/PartitionedCall_1?
$product_with_external_weights_29/addAddV2(product_with_external_weights_29/Mul:z:0;product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/add?
7product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7product_with_external_weights_29/Prod/reduction_indices?
%product_with_external_weights_29/ProdProd(product_with_external_weights_29/add:z:0@product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2'
%product_with_external_weights_29/Prods
NegNeg.product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555264

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:?????????2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:?????????2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/e?
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555254*:
_output_shapes(
&:?????????:?????????2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555638
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_5553102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555632
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_5552182
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
m
C__inference_encoder_layer_call_and_return_conditional_losses_555329

inputs
inputs_1
identity?
Dlinear_block_code_product_encoder_with_external_g_29/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_5553102F
Dlinear_block_code_product_encoder_with_external_g_29/PartitionedCall?
7differentiable_bpsk_modulation_layer_29/PartitionedCallPartitionedCallMlinear_block_code_product_encoder_with_external_g_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_55526429
7differentiable_bpsk_modulation_layer_29/PartitionedCall?
IdentityIdentity@differentiable_bpsk_modulation_layer_29/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
"__inference__traced_restore_555732
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
T
(__inference_encoder_layer_call_fn_555548
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_5552382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
+
__inference_g_555698
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namew
?
?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555235

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:?????????2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:?????????2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/e?
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555225*:
_output_shapes(
&:?????????:?????????2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
(__inference_encoder_layer_call_fn_555560
input_1
input_2
identity?
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_5553292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?1
?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555626
inputs_0
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y]
mulMulinputs_0mul/y:output:0*
T0*'
_output_shapes
:?????????2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:?????????2
sub?
&product_with_external_weights_29/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_29/Shape?
4product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????26
4product_with_external_weights_29/strided_slice/stack?
6product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_29/strided_slice/stack_1?
6product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_29/strided_slice/stack_2?
.product_with_external_weights_29/strided_sliceStridedSlice/product_with_external_weights_29/Shape:output:0=product_with_external_weights_29/strided_slice/stack:output:0?product_with_external_weights_29/strided_slice/stack_1:output:0?product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_29/strided_slice?
0product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0product_with_external_weights_29/Reshape/shape/1?
.product_with_external_weights_29/Reshape/shapePack7product_with_external_weights_29/strided_slice:output:09product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_29/Reshape/shape?
(product_with_external_weights_29/ReshapeReshapeinputs_17product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2*
(product_with_external_weights_29/Reshape?
/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_29/ExpandDims/dim?
+product_with_external_weights_29/ExpandDims
ExpandDimssub:z:08product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2-
+product_with_external_weights_29/ExpandDims?
&product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_29/stack?
%product_with_external_weights_29/TileTile4product_with_external_weights_29/ExpandDims:output:0/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2'
%product_with_external_weights_29/Tile?
/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_29/transpose/perm?
*product_with_external_weights_29/transpose	Transpose1product_with_external_weights_29/Reshape:output:08product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2,
*product_with_external_weights_29/transpose?
0product_with_external_weights_29/PartitionedCallPartitionedCall.product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_55514722
0product_with_external_weights_29/PartitionedCall?
$product_with_external_weights_29/MulMul.product_with_external_weights_29/Tile:output:09product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/Mul?
1product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_29/transpose_1/perm?
,product_with_external_weights_29/transpose_1	Transpose1product_with_external_weights_29/Reshape:output:0:product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2.
,product_with_external_weights_29/transpose_1?
2product_with_external_weights_29/PartitionedCall_1PartitionedCall0product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_55515724
2product_with_external_weights_29/PartitionedCall_1?
$product_with_external_weights_29/addAddV2(product_with_external_weights_29/Mul:z:0;product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2&
$product_with_external_weights_29/add?
7product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7product_with_external_weights_29/Prod/reduction_indices?
%product_with_external_weights_29/ProdProd(product_with_external_weights_29/add:z:0@product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2'
%product_with_external_weights_29/Prods
NegNeg.product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
+
__inference_f_555147
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namew
?
?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555668

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:?????????2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:?????????2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/e?
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555658*:
_output_shapes(
&:?????????:?????????2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
o
C__inference_encoder_layer_call_and_return_conditional_losses_555448
inputs_0
inputs_1
identity?
:linear_block_code_product_encoder_with_external_g_29/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_29/mul/y?
8linear_block_code_product_encoder_with_external_g_29/mulMulinputs_0Clinear_block_code_product_encoder_with_external_g_29/mul/y:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/mul?
:linear_block_code_product_encoder_with_external_g_29/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2<
:linear_block_code_product_encoder_with_external_g_29/sub/x?
8linear_block_code_product_encoder_with_external_g_29/subSubClinear_block_code_product_encoder_with_external_g_29/sub/x:output:0<linear_block_code_product_encoder_with_external_g_29/mul:z:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/sub?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ShapeShape<linear_block_code_product_encoder_with_external_g_29/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape?
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2k
ilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1?
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Shape:output:0rlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1?
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape?
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ReshapeReshapeinputs_1llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2_
]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim?
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_29/sub:z:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2b
`linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims?
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/TileTileilinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/stack:output:0*
T0*+
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile?
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm?
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose/perm:output:0*
T0*'
_output_shapes
:?????????2a
_linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose?
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_f_5551472g
elinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/MulMulclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Tile:output:0nlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul?
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm?
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Reshape:output:0olinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2c
alinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1?
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *
fR
__inference_g_5551572i
glinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1?
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/addAddV2]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Mul:z:0plinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/PartitionedCall_1:output:0*
T0*+
_output_shapes
:?????????2[
Ylinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add?
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2n
llinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices?
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/ProdProd]linear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/add:z:0ulinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2\
Zlinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod?
8linear_block_code_product_encoder_with_external_g_29/NegNegclinear_block_code_product_encoder_with_external_g_29/product_with_external_weights_29/Prod:output:0*
T0*'
_output_shapes
:?????????2:
8linear_block_code_product_encoder_with_external_g_29/Neg?
<linear_block_code_product_encoder_with_external_g_29/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_29/Neg:y:0*
T0*'
_output_shapes
:?????????2>
<linear_block_code_product_encoder_with_external_g_29/Sigmoid?
-differentiable_bpsk_modulation_layer_29/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_29/sub/y?
+differentiable_bpsk_modulation_layer_29/subSub@linear_block_code_product_encoder_with_external_g_29/Sigmoid:y:06differentiable_bpsk_modulation_layer_29/sub/y:output:0*
T0*'
_output_shapes
:?????????2-
+differentiable_bpsk_modulation_layer_29/sub?
1differentiable_bpsk_modulation_layer_29/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_29/Greater/y?
/differentiable_bpsk_modulation_layer_29/GreaterGreater/differentiable_bpsk_modulation_layer_29/sub:z:0:differentiable_bpsk_modulation_layer_29/Greater/y:output:0*
T0*'
_output_shapes
:?????????21
/differentiable_bpsk_modulation_layer_29/Greater?
2differentiable_bpsk_modulation_layer_29/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/t?
2differentiable_bpsk_modulation_layer_29/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2differentiable_bpsk_modulation_layer_29/SelectV2/e?
0differentiable_bpsk_modulation_layer_29/SelectV2SelectV23differentiable_bpsk_modulation_layer_29/Greater:z:0;differentiable_bpsk_modulation_layer_29/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_29/SelectV2/e:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/SelectV2?
0differentiable_bpsk_modulation_layer_29/IdentityIdentity9differentiable_bpsk_modulation_layer_29/SelectV2:output:0*
T0*'
_output_shapes
:?????????22
0differentiable_bpsk_modulation_layer_29/Identity?
1differentiable_bpsk_modulation_layer_29/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_29/SelectV2:output:0/differentiable_bpsk_modulation_layer_29/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555438*:
_output_shapes(
&:?????????:?????????23
1differentiable_bpsk_modulation_layer_29/IdentityN?
IdentityIdentity:differentiable_bpsk_modulation_layer_29/IdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
d
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555678

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_5552642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
T
(__inference_encoder_layer_call_fn_555554
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_5553292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
+
__inference_g_555692
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xM
subSubsub/x:output:0w*
T0*
_output_shapes

:2
subR
IdentityIdentitysub:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::A =

_output_shapes

:

_user_specified_namew
?
?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555653

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:?????????2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:?????????2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

SelectV2/e?
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:?????????2

Identity?
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-555643*:
_output_shapes(
&:?????????:?????????2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????4
output_1(
PartitionedCall:0?????????tensorflow/serving/predict:?`
?
	lbcpe

modulation
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*)&call_and_return_all_conditional_losses
*_default_save_signature
+__call__"?
_tf_keras_model?{"name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Encoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 16]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [16, 31]}, "float32", "input_2"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
?
product
		variables

regularization_losses
trainable_variables
	keras_api
*,&call_and_return_all_conditional_losses
-__call__"?
_tf_keras_layer?{"name": "linear_block_code_product_encoder_with_external_g_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LinearBlockCodeProductEncoderWithExternalG", "config": {"layer was saved without config": true}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*.&call_and_return_all_conditional_losses
/__call__"?
_tf_keras_layer?{"name": "differentiable_bpsk_modulation_layer_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DifferentiableBPSKModulationLayer", "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
metrics
layer_metrics

layers
layer_regularization_losses
non_trainable_variables
trainable_variables
+__call__
*_default_save_signature
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
,
0serving_default"
signature_map
?
	variables
regularization_losses
trainable_variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__
3f
4g"?
_tf_keras_layer?{"name": "product_with_external_weights_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ProductWithExternalWeights", "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 16]}, {"class_name": "TensorShape", "items": [16, 31]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
		variables

regularization_losses
layer_metrics

layers
layer_regularization_losses
metrics
non_trainable_variables
trainable_variables
-__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
layer_metrics

 layers
!layer_regularization_losses
"metrics
#non_trainable_variables
trainable_variables
/__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
$layer_metrics

%layers
&layer_regularization_losses
'metrics
(non_trainable_variables
trainable_variables
2__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
C__inference_encoder_layer_call_and_return_conditional_losses_555404
C__inference_encoder_layer_call_and_return_conditional_losses_555448
C__inference_encoder_layer_call_and_return_conditional_losses_555492
C__inference_encoder_layer_call_and_return_conditional_losses_555536?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_555176?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_1?????????
!?
input_2?????????
?2?
(__inference_encoder_layer_call_fn_555542
(__inference_encoder_layer_call_fn_555548
(__inference_encoder_layer_call_fn_555554
(__inference_encoder_layer_call_fn_555560?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555593
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555626?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555632
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555638?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555653
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555668?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555673
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555678?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_555360input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_f_555682
__inference_f_555686?
???
FullArgSpec
args?
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_g_555692
__inference_g_555698?
???
FullArgSpec
args?
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_555176?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "3?0
.
output_1"?
output_1??????????
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555653\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
c__inference_differentiable_bpsk_modulation_layer_29_layer_call_and_return_conditional_losses_555668\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555673O3?0
)?&
 ?
inputs?????????
p 
? "???????????
H__inference_differentiable_bpsk_modulation_layer_29_layer_call_fn_555678O3?0
)?&
 ?
inputs?????????
p
? "???????????
C__inference_encoder_layer_call_and_return_conditional_losses_555404?^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 
? "%?"
?
0?????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_555448?^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p
? "%?"
?
0?????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_555492?\?Y
R?O
I?F
!?
input_1?????????
!?
input_2?????????
p 
? "%?"
?
0?????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_555536?\?Y
R?O
I?F
!?
input_1?????????
!?
input_2?????????
p
? "%?"
?
0?????????
? ?
(__inference_encoder_layer_call_fn_555542x\?Y
R?O
I?F
!?
input_1?????????
!?
input_2?????????
p 
? "???????????
(__inference_encoder_layer_call_fn_555548z^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 
? "???????????
(__inference_encoder_layer_call_fn_555554z^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p
? "???????????
(__inference_encoder_layer_call_fn_555560x\?Y
R?O
I?F
!?
input_1?????????
!?
input_2?????????
p
? "??????????L
__inference_f_5556824!?
?
?
w
? "?^
__inference_f_555686F*?'
 ?
?
w?????????
? "??????????L
__inference_g_5556924!?
?
?
w
? "?^
__inference_g_555698F*?'
 ?
?
w?????????
? "???????????
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555593?^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 
? "%?"
?
0?????????
? ?
p__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_and_return_conditional_losses_555626?^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p
? "%?"
?
0?????????
? ?
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555632z^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 
? "???????????
U__inference_linear_block_code_product_encoder_with_external_g_29_layer_call_fn_555638z^?[
T?Q
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p
? "???????????
$__inference_signature_wrapper_555360?i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"3?0
.
output_1"?
output_1?????????