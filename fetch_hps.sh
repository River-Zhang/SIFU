urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


# # download mesh_downsampling file
# mkdir -p data/HPS/pymaf_data && cd data/HPS/pymaf_data/
# wget https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz

# # Model constants etc from https://github.com/nkolot/SPIN/blob/master/fetch_data.sh
# wget http://visiondata.cis.upenn.edu/spin/data.tar.gz
# tar xvf data.tar.gz
# mv data/* .
# rm -rf data && rm -f data.tar.gz

# # PyMAF pre-trained model
# gdown https://drive.google.com/drive/u/1/folders/1CkF79XRaZzdRlj6eJUt4W0nbTORv2t7O -O pretrained_model --folder
# cd ../../..
# echo "PyMAF done!"



function download_pixie(){

  mkdir -p data/HPS/pixie_data

  # SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
  echo -e "\nYou need to login https://icon.is.tue.mpg.de/ and register SMPL-X and PIXIE"
  read -p "Username (SMPL-X):" username
  read -p "Password (SMPL-X):" password
  username=$(urle $username)
  password=$(urle $password)
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/HPS/pixie_data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

  # PIXIE pretrained model and utilities
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/HPS/pixie_data/pixie_model.tar' --no-check-certificate --continue
  wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/HPS/pixie_data/utilities.zip' --no-check-certificate --continue
  cd data/HPS/pixie_data
  unzip utilities.zip
  rm utilities.zip
  cd ../../../
}

function download_hybrik(){
    mkdir -p data/HPS/hybrik_data

    # (optional) download HybrIK
    # gdown https://drive.google.com/uc?id=16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9 -O data/hybrik_data/pretrained_w_cam.pth
    gdown https://drive.google.com/uc?id=1lEWZgqxiDNNJgvpjlIXef2VuxcGbtXzi -O data/HPS/hybrik_data.zip
    cd data/HPS
    unzip hybrik_data.zip
    rm -r *.zip __MACOSX
    cd ../../

    echo "HybrIK done!"
}


read -p "(optional) Download PIXIE[SMPL-X] (y/n)?" choice
case "$choice" in 
  y|Y ) download_pixie;;
  n|N ) echo "PIXIE Done!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac

pwd
read -p "(optional) Download HybrIK[SMPL] (y/n)?" choice
case "$choice" in 
  y|Y ) download_hybrik;;
  n|N ) echo "HybrIK Done!";;
  * ) echo "Invalid input! Please use y|Y or n|N";;
esac
