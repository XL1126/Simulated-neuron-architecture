try:
    from core_cpp import NeuronPopulation, STDPConfig
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    NeuronPopulation = None
    STDPConfig = None

from .input_layer import InputLayer
from .primary_layer import PrimaryLayer
from .core_layer import CoreLayer
from .memory_layer import MemoryLayer
from .output_layer import OutputLayer, SNNDecoder

__all__ = [
    'NeuronPopulation', 'STDPConfig', 'CPP_AVAILABLE',
    'InputLayer', 'PrimaryLayer', 'CoreLayer', 'MemoryLayer',
    'OutputLayer', 'SNNDecoder',
]
