®
á
B
AssignVariableOp
resource
value"dtype"
dtypetype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
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
¿
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
H
ShardedFilename
basename	
shard

num_shards
filename
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ù
³
-AE_GNBP_0/code_generator/redundancy_weights_GVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*>
shared_name/-AE_GNBP_0/code_generator/redundancy_weights_G
¬
AAE_GNBP_0/code_generator/redundancy_weights_G/Read/ReadVariableOpReadVariableOp-AE_GNBP_0/code_generator/redundancy_weights_G*
_output_shapes	
:ð*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ö
valueÌBÉ BÂ
|
redundancy_weights_G
	variables
regularization_losses
trainable_variables
	keras_api

signatures
rp
VARIABLE_VALUE-AE_GNBP_0/code_generator/redundancy_weights_G/redundancy_weights_G/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
	variables
regularization_losses
metrics
layer_metrics

	layers

layer_regularization_losses
non_trainable_variables
trainable_variables
 
 
 
 
 
 
r
serving_default_input_1Placeholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-AE_GNBP_0/code_generator/redundancy_weights_G*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_473121
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
â
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAAE_GNBP_0/code_generator/redundancy_weights_G/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_473336
É
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename-AE_GNBP_0/code_generator/redundancy_weights_G*
Tin
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_473349È
Ø"
´
J__inference_code_generator_layer_call_and_return_conditional_losses_473018

inputs&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-472983*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú"
µ
J__inference_code_generator_layer_call_and_return_conditional_losses_473235
input_1&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-473200*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
É
ª
"__inference__traced_restore_473349
file_prefixM
>assignvariableop_ae_gnbp_0_code_generator_redundancy_weights_g:	ð

identity_2¢AssignVariableOpÕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*b
valueYBWB/redundancy_weights_G/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesµ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity½
AssignVariableOpAssignVariableOp>assignvariableop_ae_gnbp_0_code_generator_redundancy_weights_gIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: : 2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
þ

$__inference_signature_wrapper_473121
input_1
unknown:	ð
identity

identity_1¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_4729202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¯

/__inference_code_generator_layer_call_fn_473300

inputs
unknown:	ð
identity

identity_1¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_code_generator_layer_call_and_return_conditional_losses_4730182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

/__inference_code_generator_layer_call_fn_473309
input_1
unknown:	ð
identity

identity_1¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_code_generator_layer_call_and_return_conditional_losses_4730182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ì
º
__inference__traced_save_473336
file_prefixL
Hsavev2_ae_gnbp_0_code_generator_redundancy_weights_g_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÏ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*b
valueYBWB/redundancy_weights_G/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Hsavev2_ae_gnbp_0_code_generator_redundancy_weights_g_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes
: :ð: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:ð:

_output_shapes
: 
.
¨
!__inference__wrapped_model_472920
input_15
&code_generator_readvariableop_resource:	ð
identity

identity_1¢code_generator/ReadVariableOp¢
code_generator/ReadVariableOpReadVariableOp&code_generator_readvariableop_resource*
_output_shapes	
:ð*
dtype02
code_generator/ReadVariableOpy
code_generator/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
code_generator/Greater/y«
code_generator/GreaterGreater%code_generator/ReadVariableOp:value:0!code_generator/Greater/y:output:0*
T0*
_output_shapes	
:ð2
code_generator/Greater{
code_generator/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
code_generator/SelectV2/t{
code_generator/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
code_generator/SelectV2/eÈ
code_generator/SelectV2SelectV2code_generator/Greater:z:0"code_generator/SelectV2/t:output:0"code_generator/SelectV2/e:output:0*
T0*
_output_shapes	
:ð2
code_generator/SelectV2
code_generator/IdentityIdentity code_generator/SelectV2:output:0*
T0*
_output_shapes	
:ð2
code_generator/Identityé
code_generator/IdentityN	IdentityN code_generator/SelectV2:output:0%code_generator/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-472885*"
_output_shapes
:ð:ð2
code_generator/IdentityN
code_generator/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
code_generator/eye/onesx
code_generator/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
code_generator/eye/diag/k
 code_generator/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 code_generator/eye/diag/num_rows
 code_generator/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 code_generator/eye/diag/num_cols
%code_generator/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%code_generator/eye/diag/padding_value·
code_generator/eye/diagMatrixDiagV3 code_generator/eye/ones:output:0"code_generator/eye/diag/k:output:0)code_generator/eye/diag/num_rows:output:0)code_generator/eye/diag/num_cols:output:0.code_generator/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
code_generator/eye/diag
code_generator/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
code_generator/Reshape/shape®
code_generator/ReshapeReshape!code_generator/IdentityN:output:0%code_generator/Reshape/shape:output:0*
T0*
_output_shapes

:2
code_generator/Reshape
&code_generator/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&code_generator/concatenate/concat/axisø
!code_generator/concatenate/concatConcatV2 code_generator/eye/diag:output:0code_generator/Reshape:output:0/code_generator/concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2#
!code_generator/concatenate/concat
code_generator/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
code_generator/Reshape_1/shape´
code_generator/Reshape_1Reshape!code_generator/IdentityN:output:0'code_generator/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
code_generator/Reshape_1
code_generator/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
code_generator/transpose/permµ
code_generator/transpose	Transpose!code_generator/Reshape_1:output:0&code_generator/transpose/perm:output:0*
T0*
_output_shapes

:2
code_generator/transpose
code_generator/eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
code_generator/eye_1/ones|
code_generator/eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
code_generator/eye_1/diag/k
"code_generator/eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"code_generator/eye_1/diag/num_rows
"code_generator/eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"code_generator/eye_1/diag/num_cols
'code_generator/eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'code_generator/eye_1/diag/padding_valueÅ
code_generator/eye_1/diagMatrixDiagV3"code_generator/eye_1/ones:output:0$code_generator/eye_1/diag/k:output:0+code_generator/eye_1/diag/num_rows:output:0+code_generator/eye_1/diag/num_cols:output:00code_generator/eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2
code_generator/eye_1/diag
(code_generator/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(code_generator/concatenate_1/concat/axisý
#code_generator/concatenate_1/concatConcatV2code_generator/transpose:y:0"code_generator/eye_1/diag:output:01code_generator/concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2%
#code_generator/concatenate_1/concat
IdentityIdentity*code_generator/concatenate/concat:output:0^code_generator/ReadVariableOp*
T0*
_output_shapes

:2

Identity

Identity_1Identity,code_generator/concatenate_1/concat:output:0^code_generator/ReadVariableOp*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2>
code_generator/ReadVariableOpcode_generator/ReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¯

/__inference_code_generator_layer_call_fn_473291

inputs
unknown:	ð
identity

identity_1¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_code_generator_layer_call_and_return_conditional_losses_4729622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú"
µ
J__inference_code_generator_layer_call_and_return_conditional_losses_473273
input_1&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-473238*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ø"
´
J__inference_code_generator_layer_call_and_return_conditional_losses_473159

inputs&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-473124*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø"
´
J__inference_code_generator_layer_call_and_return_conditional_losses_473197

inputs&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-473162*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

/__inference_code_generator_layer_call_fn_473282
input_1
unknown:	ð
identity

identity_1¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_code_generator_layer_call_and_return_conditional_losses_4729622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ø"
´
J__inference_code_generator_layer_call_and_return_conditional_losses_472962

inputs&
readvariableop_resource:	ð

identity_1

identity_2¢ReadVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ð*
dtype02
ReadVariableOp[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yo
GreaterGreaterReadVariableOp:value:0Greater/y:output:0*
T0*
_output_shapes	
:ð2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/e}
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*
_output_shapes	
:ð2

SelectV2Y
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes	
:ð2

Identity­
	IdentityN	IdentityNSelectV2:output:0ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-472927*"
_output_shapes
:ð:ð2
	IdentityNa
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diago
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape/shaper
ReshapeReshapeIdentityN:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis­
concatenate/concatConcatV2eye/diag:output:0Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate/concats
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Reshape_1/shapex
	Reshape_1ReshapeIdentityN:output:0Reshape_1/shape:output:0*
T0*
_output_shapes

:2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permy
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposee

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diagx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis²
concatenate_1/concatConcatV2transpose:y:0eye_1/diag:output:0"concatenate_1/concat/axis:output:0*
N*
T0*
_output_shapes

:2
concatenate_1/concat{

Identity_1Identityconcatenate/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_1}

Identity_2Identityconcatenate_1/concat:output:0^ReadVariableOp*
T0*
_output_shapes

:2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ó
serving_default¿
7
input_1,
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ3
output_1'
StatefulPartitionedCall:03
output_2'
StatefulPartitionedCall:1tensorflow/serving/predict:õ#
¹
redundancy_weights_G
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*&call_and_return_all_conditional_losses
_default_save_signature
__call__"ã
_tf_keras_modelÉ{"name": "code_generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CodeGenerator", "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1]}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1]}, "int32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CodeGenerator"}}
<::ð2-AE_GNBP_0/code_generator/redundancy_weights_G
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
Ê
	variables
regularization_losses
metrics
layer_metrics

	layers

layer_regularization_losses
non_trainable_variables
trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
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
ê2ç
J__inference_code_generator_layer_call_and_return_conditional_losses_473159
J__inference_code_generator_layer_call_and_return_conditional_losses_473197
J__inference_code_generator_layer_call_and_return_conditional_losses_473235
J__inference_code_generator_layer_call_and_return_conditional_losses_473273´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
!__inference__wrapped_model_472920²
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *"¢

input_1ÿÿÿÿÿÿÿÿÿ
þ2û
/__inference_code_generator_layer_call_fn_473282
/__inference_code_generator_layer_call_fn_473291
/__inference_code_generator_layer_call_fn_473300
/__inference_code_generator_layer_call_fn_473309´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_473121input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ª
!__inference__wrapped_model_472920,¢)
"¢

input_1ÿÿÿÿÿÿÿÿÿ
ª "QªN
%
output_1
output_1
%
output_2
output_2½
J__inference_code_generator_layer_call_and_return_conditional_losses_473159o/¢,
%¢"

inputsÿÿÿÿÿÿÿÿÿ
p 
ª "9¢6
/¢,

0/0

0/1
 ½
J__inference_code_generator_layer_call_and_return_conditional_losses_473197o/¢,
%¢"

inputsÿÿÿÿÿÿÿÿÿ
p
ª "9¢6
/¢,

0/0

0/1
 ¾
J__inference_code_generator_layer_call_and_return_conditional_losses_473235p0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "9¢6
/¢,

0/0

0/1
 ¾
J__inference_code_generator_layer_call_and_return_conditional_losses_473273p0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "9¢6
/¢,

0/0

0/1
 
/__inference_code_generator_layer_call_fn_473282b0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+¢(

0

1
/__inference_code_generator_layer_call_fn_473291a/¢,
%¢"

inputsÿÿÿÿÿÿÿÿÿ
p 
ª "+¢(

0

1
/__inference_code_generator_layer_call_fn_473300a/¢,
%¢"

inputsÿÿÿÿÿÿÿÿÿ
p
ª "+¢(

0

1
/__inference_code_generator_layer_call_fn_473309b0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+¢(

0

1¸
$__inference_signature_wrapper_4731217¢4
¢ 
-ª*
(
input_1
input_1ÿÿÿÿÿÿÿÿÿ"QªN
%
output_1
output_1
%
output_2
output_2