import os

class FormatDict:
    def __init__(self, title, dictionary):
        self.title = title
        self.dictionary = dictionary
        self.max_key = max([len(l) for l in self.dictionary.keys()])
        self.max_val = max([len(str(l)) for l in self.dictionary.values()]) 

    def __format__(self, fmt):
        l = list(self.dictionary.items())
        s = []
        s.append('+' + '-'*(self.max_key + self.max_val + 5) + '+\n')
        s.append('|{v:^{vw}}|\n'.format(v=self.title, vw=self.max_key + self.max_val + 5))
        s.append('+' + '-'*(self.max_key+2) + '+' + '-'*(self.max_val+2) + '+\n')
        for item in l:
            s.append('| {k:<{kw}} | {v:<{vw}} |\n'.format(k=str(item[0]), 
                                        kw=self.max_key, 
                                        v=str(item[1]), 
                                        vw=self.max_val))
            s.append('+' + '-'*(self.max_key+2) + '+' + '-'*(self.max_val+2) + '+\n')
        return ''.join(s)

def log(title, **kwargs):
    print('{}'.format(FormatDict(title, kwargs)))

if __name__ == '__main__':
    d = { 'test1': 4, 'superlong': 'week', 'small': 12.234 }
    log('mytitle', **d)

