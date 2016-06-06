class LibdyniError(Exception):
    '''The root libdyni exception class'''
    pass


class ParameterError(LibdyniError):
    pass


class ParsingError(LibdyniError):
    pass

class GeneratorError(LibdyniError):
    pass
