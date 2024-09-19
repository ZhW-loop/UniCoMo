import sys
sys.path.append("tvm_app")
import tvm
from typing import Dict, List, Any, Union, Literal

from tvm.tir.schedule import Instruction, BlockRV, ExprRV, LoopRV 
from tvm.runtime.container import String
from tvm.ir.container import Array
from tvm.tir.function import IndexMap
from tvm.tir.expr import IntImm, FloatImm

def is_float(input_str):
    try:
        float_value = float(input_str)
        return True, float_value
    except ValueError:
        return False, str(input_str)

def TranslateInputRVs(inputs, rv_names: Dict[Any, str]):
    results = []
    for input in inputs:
        if input in rv_names:
            results.append(rv_names[input])
        elif isinstance(input, String) or isinstance(input, IndexMap) or input is None:
            results.append(str(input))
        elif isinstance(input, IntImm) or isinstance(input, FloatImm):
            results.append(str(input.value))
        elif isinstance(input, Array) or isinstance(input, list):
            input = list(input)
            results.extend(TranslateInputRVs(input, rv_names))
        elif isinstance(input, BlockRV) or isinstance(input, LoopRV) or isinstance(input, ExprRV):
            raise ValueError("TranslateInputRVs, Undefine RV...")
        else:
            raise ValueError("TranslateInputRVs, Unsupport attr type...")
    return results
'''
    def str_inputs(inputs):
        for i in range(len(inputs)):
            if inputs[i] in rv_names:
                inputs[i] = rv_names[inputs[i]]
            elif isinstance(inputs[i], String):
                inputs[i] = str(inputs[i])
            elif isinstance(inputs[i], IntImm) or isinstance(inputs[i], FloatImm):
                inputs[i] = str(inputs[i].value)
            elif isinstance(inputs[i], IndexMap):
                inputs[i] = str(inputs[i])
            elif isinstance(inputs[i], Array) or isinstance(inputs[i], list):
                inputs[i] = list(inputs[i])
                str_inputs(inputs[i])
            elif inputs[i] is None:
                inputs[i] = 'None'
            else:
                raise ValueError("TranslateInputRVs, Unsupport attr type...")
    def flatten_inputs(inputs):
        flattened = []
        for input in inputs:
            if isinstance(input, list) or isinstance(input, Array):
                input = list(input)
                flattened.extend(flatten_inputs(input))
            else:
                flattened.append(input)
        return flattened
    str_inputs(inputs)
    results = flatten_inputs(inputs)
    return results
'''

def TranslateDecisions(decisions):
    results = []
    if isinstance(decisions, Array) or isinstance(decisions, list):
        decisions = list(decisions)
        for decision in decisions:
            results.extend(TranslateDecisions(decision))
    elif isinstance(decisions, String) or isinstance(decisions, IndexMap) or decisions is None:
        results.append(str(decisions))
    elif isinstance(decisions, IntImm) or isinstance(decisions, FloatImm):
        results.append(str(decisions.value))
    else:
        raise ValueError("TranslateDecisions, Unsupport attr type...")
    return results

def TranslateAttrs(attrs):
    results = []
    for attr in attrs:
        if isinstance(attr, Array) or isinstance(attr, list):
            attr = list(attr)
            results.extend(TranslateAttrs(attr))
        elif isinstance(attr, String) or isinstance(attr, IndexMap) or attr is None:
            results.append(str(attr))
        elif isinstance(attr, IntImm) or isinstance(attr, FloatImm):
            results.append(str(attr.value))
        else:
            raise ValueError("TranslateAttrs, Unsupport attr type...")
    return results
    '''
    attrs = list(attrs)
    def str_attrs(attrs):
        for i in range(len(attrs)):
            if isinstance(attrs[i], Array) or isinstance(attrs[i], list):
                attrs[i] = list(attrs[i])
                str_attrs(attrs[i])
            elif isinstance(attrs[i], String):
                attrs[i] = str(attrs[i])
            elif isinstance(attrs[i], IntImm) or isinstance(attrs[i], FloatImm):
                attrs[i] = str(attrs[i].value)
            elif isinstance(attrs[i], IndexMap):
                attrs[i] = str(attrs[i])
            elif attrs[i] is None:
                attrs[i] = 'None'
            else:
                raise ValueError("TranslateAttrs, Unsupport attr type...")
    def flatten_attrs(attrs):
        flattened = []
        for attr in attrs:
            if isinstance(attr, list) or isinstance(attr, Array):
                attr = list(attr)
                flattened.extend(flatten_attrs(attr))
            else:
                flattened.append(attr)
        return flattened
    str_attrs(attrs)
    results = flatten_attrs(attrs)
    return results
    '''

def TranslateAddOutputRVs(outputs, rv_names: Dict[Any, str], decisions_list:List[str]):
    if decisions_list != ['None']:
        assert len(decisions_list) == len(outputs)
    results: List[str] = [] 
    for oidx, output in enumerate(outputs):
        i = len(rv_names)
        assert output not in rv_names
        if isinstance(output, BlockRV):
            # result = 'b' + str(i)
            result = str(i)
        elif isinstance(output, LoopRV):
            # result = 'l' + str(i)
            result = str(i)
        elif isinstance(output, ExprRV):
            # result = 'v' + str(i)
            result = decisions_list[oidx]
        results.append(result)
        rv_names[output] = result
    return results
        
    
def Sch2Listdict(sch: Union[tvm.tir.Schedule, tvm.tir.schedule.Trace]):
    if isinstance(sch, tvm.tir.Schedule):
        SchTrace:tvm.tir.schedule.Trace = sch.trace
    else:
        SchTrace = sch
    rv_names:Dict[Any, str] = {}
    results_Listdict:List[Dict[Literal['kind', 'inputs', 'attrs', 'decisions', 'outputs'], \
                                List[str]]] = []
    for i, inst in enumerate(SchTrace.insts):
        dic = {}
        dic['kind'] = [inst.kind.name]
        dic['inputs'] = TranslateInputRVs(inst.inputs, rv_names)
        dic['attrs'] = TranslateAttrs(inst.attrs)
        dic['decisions'] = TranslateDecisions(SchTrace.get_decision(inst))
        dic['outputs'] = TranslateAddOutputRVs(inst.outputs, rv_names, dic['decisions'])
        
        results_Listdict.append(dic)
    return results_Listdict
        
def print_Listdict(results_Listdict, keys=['kind', 'inputs', 'attrs', 'decisions', 'outputs']):
    for result in results_Listdict:
        for key in keys:
            print(f"{key}: {result[key]}")

if __name__ == "__main__":
    from tvm_app.common import get_MatmulModule
    from tvm_app.sch_trace_gpu import apply_trace as gpu_apply_trace
    from tvm_app.sch_trace_cpu import apply_trace as cpu_apply_trace
    
    mod = get_MatmulModule(64, 256, 128, 'float16', 'float16', 'float16', )
    sch = tvm.tir.Schedule(mod)
    gpu_apply_trace(sch)
    
    # mod = get_MatmulModule(64, 128, 1024, 'uint8', 'int8', 'int32', \
    #     True, "llvm -mcpu=cascadelake -num-cores 4")
    # sch = tvm.tir.Schedule(mod)
    # cpu_apply_trace(sch)
    results_Listdict = Sch2Listdict(sch)
    print_Listdict(results_Listdict)