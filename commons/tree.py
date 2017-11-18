import math
import operator  # wrapper of c op in python


# TODO Check implementaton 1st is openai
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            # return the value of the node
            return self._value[node]
        # cut in half
        mid = (node_start + node_end) // 2  # first iter is capacity //2
        # secpond iter mid = capacity//4: => end >= mid

        if end <= mid:
            # reapply reduce helper in from 0 to mid
            # es. node_start = 0, node_end = capacity, mid = from 0 to mid)
            return self._reduce_helper(start=start,
                                       end=end,
                                       node=2 * node,
                                       node_start=node_start,
                                       node_end=mid)
        else:
            # capacity//4 + 1 <=start reapply reduce helper from capacity//4 + 1 to capacity //2
            # at 3 iter (capacity // 4 +1  + capacity // 2 ) //2: => mid+1 > start
            #  we found the intermediete node
            if mid + 1 <= start:
                return self._reduce_helper(
                    start=start,
                    end=end,
                    node=2 * node + 1,
                    node_start=mid + 1,
                    node_end=node_end
                )
            else:
                return self._operation(
                    self._reduce_helper(start=start,
                                        end=mid,
                                        node=2 * node,
                                        node_start=node_start,
                                        node_end=mid),
                    self._reduce_helper(start=mid + 1,
                                        end=end,
                                        node=2 * node + 1,
                                        node_start=mid + 1,
                                        node_end=node_end)
                )

    def reduce(self, start=0, end=None):
        # we want:
        # operation(arr[start], operation(arr[start+1],operation(arr[start+2])))
        # thus we pick a start and end point and we compute the bottom of the subtree
        # and we pick the tuple of two adajents node and we apply the operation
        # this is done via a recurisve function that recurisvely reduce the size of the tree
        # by ((now/2 + before)/2) at each step
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, node=1, node_start=0, node_end=self._capacity)

    def __setitem__(self, idx, val):
        # set item is a bultin method for lists that replace the item at index idx with val
        # start from the n-1 val, replace withe value
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        # apply the op recursively to each op(idx, idx+1)
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        # return the highet index at which the upper bound is accepted
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while is not a leaf node
            if self._value[2 * idx] > prefixsum:
                # if the value of the 2n node is higher thant he upper bound
                # move two nodes down i.e.
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                # reduce the upper bound by the bound of the node
                # expand the current node
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = int(math.ceil(math.log(max_size + 1, 2)) + 1)
        self.tree_size = int(2 ** self.tree_level - 1)
        self.tree = [0 for _ in range(self.tree_size)]
        self.data = [None for _ in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:
            return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
                2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left:
            return self._find(value, 2 * index + 1)
        else:
            return self._find(value - left, 2 * (index + 1))

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j])

    def filled_size(self):
        return self.size


if __name__ == "__main__":
    op = operator.add

    general_tree = SegmentTree(capacity=8, operation=op, neutral_element=0.0)
    general_tree[2] = 4.
    general_tree.reduce(start=0, end=4)
    assert sum(general_tree._value) == 16

    s = SumTree(10)
    for i in range(20):
        s.add(2 ** i, i)
    s.print_tree()
    print(s.find(0.5))
