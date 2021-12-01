
fnames=()
fnames[0]=01_canny_gradient_x.png
fnames[1]=02_canny_gradient_y.png
fnames[2]=03_canny_gradient.png
fnames[3]=04_canny_angles.png
fnames[4]=05_canny_non_maxima_supression.png
fnames[5]=06_canny_end_result.png
fnames[6]=07_dilation.png
fnames[7]=08_erosion.png
fnames[8]=09_hough_accumulator_circles.png
fnames[9]=10_hough_circles_own.png
fnames[10]=11_hough_accumulator_lines.png
fnames[11]=12_hough_lines_local_maximums.png
fnames[12]=13_hough_lines_own.png
fnames[13]=14_hough_lines_default.png


for tc in "coins1" "coins3" "coins_test_3"
do

    mkdir -p dif/${tc}
    for fname in ${fnames[*]}
    do
        convert data/ref_x64/${tc}/${fname} output/${tc}/${fname} -compose difference -composite -negate -contrast-stretch 0 dif/${tc}/${fname}
    done
done


