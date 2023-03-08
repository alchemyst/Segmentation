import segment

data = segment.DataContainer.fromfile('testdata/almostlinear.dat')

def test_bottom_up():
    s = segment.BottomUp(segment.LinearRegression, 2)
    s.segment(data)


def test_top_down():
    s = segment.TopDown(segment.LinearRegression, 2)
    s.segment(data)

