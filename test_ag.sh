# GRO Tests
echo "R=2 Models - GRO" > "2.txt"
echo "Blind No DC" >> "2.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind_no_dc --nodc --mask-type 1 --R 2 >> "2.txt"

echo "R=4 Models - GRO" > "4.txt"
echo "Blind No DC" >> "4.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind_no_dc --nodc --mask-type 1 --R 4 >> "4.txt"

echo "R=6 Models - GRO" > "6.txt"
echo "Blind No DC" >> "6.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind_no_dc --nodc --mask-type 1 --R 6 >> "6.txt"

#echo "Baseline UNet" >> "4.txt"
#python test.py --mri --exp-name neurips/l1_ssim_agnostic --mask-type 1 --R 4 >> "4.txt"
#
#echo "Random VarNet" >> "4.txt"
#python test_varnet.py --mri --mask-type 1 --exp-name neurips/e2e_varnet_agnostic --R 4 >> "4.txt"
#
echo "R=8 Models - GRO" > "8.txt"
echo "Blind No DC" >> "8.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind_no_dc --nodc --mask-type 1 --R 8 >> "8.txt"

#echo "Baseline UNet" >> "8.txt"
#python test.py --mri --exp-name neurips/l1_ssim_agnostic --mask-type 1 --R 8 >> "8.txt"

#echo "Random VarNet" >> "8.txt"
#python test_varnet.py --mri --mask-type 1 --exp-name neurips/e2e_varnet_agnostic --R 8 >> "8.txt"

## Random tests
#echo "Random Test" > "random.txt"
#echo "Proposed" >> "random.txt"
#python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_proposed--mask-type 4 --R $i >> "random.txt"
#
#echo "Baseline UNet" >> "random.txt"
#python test.py --mri --exp-name neurips/l1_ssim_agnostic --mask-type 4 --R $i >> "random.txt"
#
#echo "Random VarNet" >> "random.txt"
#python test_varnet.py --mri --mask-type 4 --exp-name neurips/e2e_varnet_agnostic --R $i >> "random.txt"