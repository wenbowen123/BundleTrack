def main():
    dataset_name_lsit = ['brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25',
        'brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25',
        'brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25',
        'brown_cogsci_8---brown_cogsci_8---skip-10-dilate-25',
        'brown_cs_3---brown_cs3---skip-10-dilate-25',
        'brown_cs_7---brown_cs7---skip-10-dilate-25',
        'buckingham_palace',
        'fountain',
        'gms-large-cabinet',
        'gms-teddy',
        'harvard_c10---hv_c10_2---skip-10-dilate-25',
        'harvard_c4---hv_c4_1---skip-10-dilate-25',
        'harvard_conf_big---hv_conf_big_1---skip-10-dilate-25',
        'harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25',
        'harvard_robotics_lab---hv_s1_2---skip-10-dilate-25',
        'herzjesu',
        'home_ac---home_ac_scan1_2012_aug_22---skip-10-dilate-25',
        'hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25',
        'mit_32_g725---g725_1---skip-10-dilate-25',
        'mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25',
        'mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25',
        'mit_w85g---g_0---skip-10-dilate-25',
        'mit_w85h---h2_1---skip-10-dilate-25',
        'notre_dame_front_facade',
        'reichstag',
        'sacre_coeur',
        'st_peters_square',
    ]
    ds_type_list = ['train', 'test', 'val']
    in_dir_base = './datasets/{0}/{1}/images'
    out_dir_base = './outputs_lfnet_feat/{0}/{1}'
    base_str = 'python ./run_lfnet.py --in_dir={0} --out_dir={1} --max_longer_edge=-1'
    for dataset_name in dataset_name_lsit:
        for ds_type in ds_type_list:
            in_dir = in_dir_base.format(dataset_name, ds_type)
            out_dir = out_dir_base.format(dataset_name, ds_type)
            command = base_str.format(in_dir, out_dir)
            print(command)
            # with open('./jobs/todo/{0}.sh'.format(dataset_name), 'w') as f:
                # f.write("#!/bin/bash\n")
                # f.write("source ~/.bashrc\n")
                # f.write("activate torch1.0\n")
                # f.write('{0}\n'.format(command))

if __name__ == '__main__':
    main()
