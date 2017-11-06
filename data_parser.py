import xlrd
import numpy as np


def is_header(row, template):
    for (value, cell) in zip(template, row):
        if value != cell.value:
            return False
    return True


def row_to_array(row):
    return list(map(lambda cell: float(cell.value) if cell.value else cell.value, row))


ARSNOC_HEADER = ['a', 'x', 'y']


def parse_arsnoc():
    def parse_arsnoc_sheet(sheet):
        track = []

        is_first = None

        for pos in range(sheet.nrows):
            row = sheet.row(pos)
            if is_first is None:
                if is_header(row, ARSNOC_HEADER):
                    is_first = True
            else:
                row = row_to_array(row[:3])
                if is_first:
                    r = row[0]
                    is_first = False
                if type(row[1]) is float:
                    track.append([r] + row[1:])

        return np.array(track)

    def parse_arsnoc_book(book):
        result = []
        for sheet in book.sheets():
            track = parse_arsnoc_sheet(sheet)
            if track.shape[0] != 0:
                result.append(track)

        return result

    data = []
    for i in range(2, 15):
        file_name = './data/arsnoc{}cnfoc.xls'.format(i)
        book = xlrd.open_workbook(file_name)
        data.extend(parse_arsnoc_book(book))

    return data


def parse_new_tracks():
    HEADER   = ['клетка 1', "Area", "XM", "YM", "XStart", "YStart"]
    NEW_CELL = ['клетка 2', '', '', '', '', '']

    def is_empty(row):
        for cell in row:
            if cell.value:
                return False
        return True

    def is_new_cell(row):
        for i, cell in enumerate(row):
            if NEW_CELL[i] != cell.value:
                return False
        return True

    def parse_sheet(sheet):
        data = []
        track = []

        assert is_header(sheet.row(0), HEADER)

        for pos in range(1, sheet.nrows):
            row = sheet.row(pos)
            if is_empty(row):
                if track:
                    data.append(np.array(track))
                    track = []
            elif is_new_cell(row):
                if track:
                    data.append(np.array(track))
                    track = []
            else:
                track.append(row_to_array(row))

        if track:
            data.append(np.array(track))

        return data

    def convert_new_track(track):
        return track[:, 1:4]

    def parse_new_tracks_book(book):
        data = []
        for sheet in book.sheets():
            data.extend(parse_sheet(sheet))
        return list(map(convert_new_track, data))

    return parse_new_tracks_book(xlrd.open_workbook('./data/tracks_new.xls'))


def convert_area_to_radius(track):
    import math
    track[:, 0] = np.sqrt(track[:, 0] / math.pi)
    return track


def convert_to_normal_coords(track):
    return track * 0.05 * 1e-6


def read_data():
    data = parse_arsnoc() + parse_new_tracks()
    return list(map(convert_to_normal_coords, map(convert_area_to_radius, data)))


if __name__ == "__main__":
    data = read_data()
    print(data[0])
