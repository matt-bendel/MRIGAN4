# GRO Tests
for ((i=2; i<9; i++)); do
  echo "R=$i Models - GRO" > "$i.txt"
  echo "Proposed" >> "$i.txt"
  python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_proposed--mask-type 1 --R $i >> "$i.txt"

  echo "Proposed - No DC" >> "$i.txt"
  python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_proposed_no_dc--mask-type 1 --R $i --nodc >> "$i.txt"

  echo "Blind" >> "$i.txt"
  python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind--mask-type 1 --R $i >> "$i.txt"

  echo "Baseline UNet" >> "$i.txt"
  python test.py --mri --exp-name neurips/l1_ssim_agnostic --mask-type 1 --R $i >> "$i.txt"

  echo "Random VarNet" >> "$i.txt"
  python test_varnet.py --mri --mask-type 1 --exp-name neurips/e2e_varnet_agnostic --R $i >> "$i.txt"

  if [$i == 4]; then
    echo "R=4 VarNet" >> "$i.txt"
    python test_varnet.py --mri --mask-type 5 --exp-name neurips/e2e_varnet_R=4 --R $i >> "$i.txt"

    echo "R=4 rcGAN" >> "$i.txt"
    python test.py --mri --exp-name neurips/rcgan_R=4 --mask-type 1 --R $i >> "$i.txt"
  fi
done

# Random tests
echo "Random Test" > "random.txt"
echo "Proposed" >> "random.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_proposed--mask-type 4 --R $i >> "random.txt"

echo "Proposed - No DC" >> "random.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_proposed_no_dc--mask-type 4 --R $i --nodc >> "random.txt"

echo "Blind" >> "random.txt"
python test.py --mri --rcgan --exp-name neurips/rcgan_agnostic_blind--mask-type 4 --R $i >> "random.txt"

echo "Baseline UNet" >> "random.txt"
python test.py --mri --exp-name neurips/l1_ssim_agnostic --mask-type 4 --R $i >> "random.txt"

echo "Random VarNet" >> "random.txt"
python test_varnet.py --mri --mask-type 4 --exp-name neurips/e2e_varnet_agnostic --R $i >> "random.txt"