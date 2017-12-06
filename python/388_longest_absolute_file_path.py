'''
388. Longest Absolute File Path
'''

# first naive approach
class Solution(object):

    def normalize_string(self, input):
        rpath = input
        rpath = rpath.replace('\n', '\\n')
        rpath = rpath.replace('\t', '\\t')
        # can't just replace all whitespaces as filenames and directories can
        # contain spaces
        # while iterating through each component, remove prefixes ('\n', '\t'),
        # then strip the remaining string
        return rpath

    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """

        normalized_path = self.normalize_string(input)
        longest_pathname = 0
        pathname = [0] * 100
        elems = list(normalized_path.strip().split("\\"))
        t_count = 0

        for i, elem in enumerate(elems):
            component = elem.strip()
            if i == 0 and "." not in component:
                pathname[0] = len(component)
            elif component == "n":
                t_count = 0
            elif component == "t":
                t_count += 1
            elif "." in component:
                #
                if i == 0:
                    return len(component)
                ncomp = list(component[1:])
                for i in range(max(4, len(ncomp))):
                    if ncomp[i] == ' ':
                        ncomp[i] = ''
                    elif ncomp[i] != ' ':
                        break
                ncomp = "".join(ncomp)
                print(ncomp)
                t_count += 1
                pathname_length = 0
                for i in range(t_count):
                    pathname_length += pathname[i] + 1
                pathname_length += len(ncomp)
                longest_pathname = max(pathname_length, longest_pathname)
            else:
                # in this case, it is a directory with an 'n' or 't' prefixed
                ncomp = component[1:].strip()
                t_count += 1
                pathname[t_count] = len(ncomp)

        return longest_pathname

# solution
class Solution(object):

    def lengthLongestPath(self, input):
        maxlen = 0
        # using dictionary since unknown length of paths
        pathlen = {0: 0}
        # split into components (separated by new lines)
        for line in input.splitlines():
            # strip out the tabs on the left
            name = line.lstrip('\t')
            # find the 'depth' of the dir or file by determining # of tabs
            # stripped
            depth = len(line) - len(name)
            # if it's a file then
            if '.' in name:
                # update max length with current depth path length and
                # length of the filename
                maxlen = max(maxlen, pathlen[depth] + len(name))
            else:
                # otherwise update the path length for the subsequent depth
                # with the name of
                # the current path and one extra for a slash character
                pathlen[depth + 1] = pathlen[depth] + len(name) + 1
        return maxlen


