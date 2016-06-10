import bisect

class Match:
    def __init__(self, allowSameNeighbor = False, first = None):
        self.allowSameNeighbor = allowSameNeighbor
        self.first = first
        if(self.first):
            self.dataList = [first]
        else:
            self.dataList = []

    def __len__(self):
        return len(self.dataList)

    def __bool__(self):
        return bool(self.dataList)

    def __getitem__(self, i):
        return self.dataList[i]

    def __setitem__(self, i, v):
        self.dataList[i] = v
        self.clean()

    def add(self, x, y):
        if(not self.first is None):
            assert(x >= self.first[0])
        if(len(self.dataList) == 0):
            self.dataList.append((x, y))
            return
        i = bisect.bisect_left(next(zip(*self.dataList)), x)
        if(i >= len(self.dataList)):
            if(self.allowSameNeighbor or self.dataList[-1][1] != y):
                self.dataList.append((x, y))
        elif(self.dataList[i][0] == x):
            if(self.dataList[i][1] != y):
                self.dataList[i] = (x, y) # change
                self.clean() # maybe remove
        else:
            if(self.allowSameNeighbor or self.dataList[i][1] != y):
                self.dataList.insert(i, (x, y)) # insert
            else:
                self.dataList[i] = (x, y) # move
                return

    def remove(self, idx):
        if(idx == 0 and not self.first is None):
            self.dataList[0] = self.first
        else:
            del self.dataList[idx]
        self.clean()

    def clean(self):
        if(not self.allowSameNeighbor):
            while True:
                same = []
                for idx, (x, y) in enumerate(self.dataList[:-1]):
                    if(y == self.dataList[idx + 1][1]):
                        same.append(idx + 1)
                if(same):
                    for idx in reversed(same):
                        del self.dataList[idx]
                else:
                    break

    def query(self, x):
        return self.queryCore(next(zip(*self.dataList)), x)

    @staticmethod
    def queryCore(xList, x):
        if(len(xList) == 1):
            return 0
        elif(x >= xList[-1]):
            return len(xList) - 1
        i = max(0, bisect.bisect_right(xList, x) - 1)
        return i
