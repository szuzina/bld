from data_downloader import DataDownloader
from dataloader import DataLoader, check_contours_on_slice
from msi_calculator import MSICalculator


def main():
    folder_url_ref = 'https://drive.google.com/uc?export=download&id=1Rw-MatnpUFDucEkEAZDbmAo_nSaOUD4T'
    folder_url_test = 'https://drive.google.com/uc?export=download&id=1hjiy-CBrHN1Gr2pGUJ8hnVwm-3TwUVWI'

    DataDownloader(ref_url=folder_url_ref, test_url=folder_url_test,
                   data_folder="data")

    number = 1
    dl = DataLoader(data_folder="data", number=number)

    im_slice = 'slice100'
    points_ref = dl.c_ref[im_slice]
    points_test = dl.c_test[im_slice]
    check_contours_on_slice(test_points=points_test,
                            ref_points=points_ref)

    IL_CONST = 1
    OL_CONST = 1

    msi_calc = MSICalculator(
        il=IL_CONST, ol=OL_CONST,
        ref_points=points_ref, test_points=points_test)
    msi_calc.run()

    print("The value of the MSI corresponding the selected slice is ", msi_calc.msi)


if __name__ == '__main__':
    main()
