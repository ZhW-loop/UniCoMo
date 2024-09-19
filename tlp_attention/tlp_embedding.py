from sch_trace_process import is_float
import os
from embedding_table import get_embedding_table
from typing import Any, List, Dict, Union

class Embedding:
    
    def __init__(self, 
                 measured_records: str,
                 tlp_embedding_table) -> None:
        
        if not os.path.exists(os.path.join(measured_records, tlp_embedding_table)):
            raise ValueError("make tlp embedding table first ...")
        self.kind_embedding, self.str_embedding, self.max_embedding_len, self.max_seq_len = \
            get_embedding_table(measured_records, tlp_embedding_table)
    
    def token_embedding(self, token: str) -> Union[float, List]:
        result = None
        first, second = is_float(token)
        if first:
            result = second
        elif second in self.kind_embedding:
            result = self.kind_embedding[second] # List 
        elif second in self.str_embedding:
            # use this will have high accuracy, 
            # but make tlp heavily depend on token embedding table
            # cannot align with search based test
            # result = self.str_embedding[second] 
            
            # use this is normal case, 
            result = 1.
        else:
            # raise ValueError(f"can not embedding {token} ...")
            result = 1.
        assert result is not None
        return result
        
    
    def __call__(self, token: str) -> Union[float, List]:
        return self.token_embedding(token)

    def __str__(self) -> str:
        return f"{self.kind_embedding}\n{self.str_embedding}\n{self.max_embedding_len}\n{self.max_seq_len}\n"
        