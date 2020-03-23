from math import ceil, log10

def sigfig(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - ceil(log10(abs(x)))))