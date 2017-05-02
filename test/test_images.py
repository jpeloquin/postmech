from mechana.images import decode_impath

def test_decode_impath_tiff():
    pth = 'cam0_12345_10.1.tiff'
    expected = {'Camera ID': '0',
                'Frame ID': '12345',
                'Timestamp (s)': 10.1}
    actual = decode_impath(pth)
    for k in expected:
        assert actual[k] == expected[k]

def test_decode_impath_csv():
    pth = 'cam0_12345_10.1.csv'
    expected = {'Camera ID': '0',
                'Frame ID': '12345',
                'Timestamp (s)': 10.1}
    actual = decode_impath(pth)
    for k in expected:
        assert actual[k] == expected[k]
