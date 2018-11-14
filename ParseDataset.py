import numpy as np

class ParseDataset(object):
    data = None
    data_class = None
    entry_labels = None
    attributes = None

    def __init__(self, path, separator_att = ';', separator_entries = ':'):
        file = open(path)
        lines = file.readlines()
        type = lines[0].rstrip()
        num_rows = int(lines[1])
        num_cols = int(lines[2])
        self.attributes = np.array(lines[3].rstrip().split(separator_att))
        self.data = np.zeros((num_rows, num_cols))
        self.entry_labels = [''] * num_rows
        self.data_class = np.zeros((num_rows), dtype = np.int32)
        # attributes = attributes[:-1]

        # print(self.attributes, len(self.attributes), (num_rows, num_cols))
        for index, line in enumerate(lines[4:]):
            line = line.rstrip()
            raw_entry = line.split(separator_att)
            label = raw_entry[0]
            point = raw_entry[1:- 1]

            if (type == 'SY'):

                for idx, dim in enumerate(point):
                    pair = dim.split(separator_entries)

                    # print('UHUL:', int(pair[0]))
                    self.data[index, int(pair[0])] = float(pair[1])
            elif (type == 'DY'):
                self.data[index:] = [float(i) for i in point]

            self.entry_labels[index] = label

            # print(self.entry_labels[index])
            self.data_class[index] = int(float(raw_entry[-1]))

# ParseDataset('/home/evarildo/UnB/TG1/Dataset/noticias.data')
# ParseDataset('/home/evarildo/UnB/TG1/Dataset/ImagensCorel.data')


