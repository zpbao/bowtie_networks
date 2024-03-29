python task_launcher.py \
--ngf=256  \
--ndf=128  \
--z_dim=128 \
--update_g_every=1 \
--lamb=5.0  \
--resume="auto"  \
--name="cars-lamb5"  \
--trial_id="TEST" \
--beta1=0.5 \
--save_path="./cars" \
--angles="[-90,90,-90,90,-90,90]" \
--epochs=1200 \
--batch_size=32 \
--save_every=200 \
--sample_num=8 \
--lamda=1.0 \
--data_path="./cars-64.npz" \
--num_classes=360 \


python task_launcher.py \
--ngf=256  \
--ndf=128  \
--z_dim=128 \
--update_g_every=1 \
--lamb=5.0  \
--resume="auto"  \
--name="cars-lamb5"  \
--trial_id="TEST" \
--beta1=0.5 \
--save_path="./cars" \
--angles="[-90,90,-90,90,-90,90]" \
--epochs=1400 \
--batch_size=32 \
--save_every=50 \
--sample_num=5 \
--few_shot \
--lamda=1.0 \
--data_path="./cars-64.npz" \
--num_classes=360 \
