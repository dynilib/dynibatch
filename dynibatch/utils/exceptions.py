class DynibatchError(Exception):
    '''The root libdyni exception class'''
    pass


class ParameterError(DynibatchError):
    pass


class ParsingError(DynibatchError):
    pass


class GeneratorError(DynibatchError):
    pass
