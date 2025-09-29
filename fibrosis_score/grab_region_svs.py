from pathlib import Path
import openslide
from PIL import Image
from matplotlib import pyplot as plt

# image to visualize
svs_path = Path("/home/riware/Desktop/mittal_lab/kpmp_regions/svs/daea1032-d81b-45d3-8d26-831c141d2cab_S-2010-012855_PAS_1of2.svs")
svs_folder = svs_path.parent
svs_name = svs_path.stem
identifier = "1_region"

def readsvs(svs_path: Path, level: int, location: tuple, size: tuple):
    slide = openslide.OpenSlide(svs_path)
    img = slide.read_region(location=location, level=level, size=size)
    return img

# left/right, up/down 

# region selection for glom training 
# img = readsvs(svs_path, 0, (18500,14000), (3000, 3000)) # 35e623b2-c9e8-4098-85fb-1489d04fc41d_S-2006-003982_TRI_2of2.svs region 0 
# img = readsvs(svs_path, 0, (19500,19000), (3000, 3000)) # 35e623b2-c9e8-4098-85fb-1489d04fc41d_S-2006-003982_TRI_2of2.svs region 1 
# img = readsvs(svs_path, 0, (125500,9500), (3000, 3000)) # a57cc3d6-b4b8-45d5-ad1c-db1d535c5bd5_S-2301-008479_SIL_1of2.svs region 0 
# img = readsvs(svs_path, 0, (34000,25500), (10000, 10000)) # a57cc3d6-b4b8-45d5-ad1c-db1d535c5bd5_S-2301-008479_SIL_1of2.svs region 1 
# img = readsvs(svs_path, 0, (38000,28000), (3000, 3000)) # a57cc3d6-b4b8-45d5-ad1c-db1d535c5bd5_S-2301-008479_SIL_1of2.svs region 2 
# img = readsvs(svs_path, 0, (4000,3500), (3000, 3000)) # 6aea6322-b49c-4d07-bc72-34c56f081665_S-2311-014796_HE_2of2.svs region 0 
# img = readsvs(svs_path, 0, (9000,8500), (3000, 3000)) # 6aea6322-b49c-4d07-bc72-34c56f081665_S-2311-014796_HE_2of2.svs region 1
# img = readsvs(svs_path, 0, (11000,8000), (3000, 3000)) # 6aea6322-b49c-4d07-bc72-34c56f081665_S-2311-014796_HE_2of2.svs region 2
# img = readsvs(svs_path, 0, (16500,15000), (3000, 3000)) # 6aea6322-b49c-4d07-bc72-34c56f081665_S-2311-014796_HE_2of2.svs region 3
# img = readsvs(svs_path, 0, (15500,24500), (3000, 3000)) # 12ffc235-df5d-4c01-ad13-af627e3f9146_S-2306-012864_HE_2of2.svs region 0
# img = readsvs(svs_path, 0, (16500,31000), (3000, 3000)) # 12ffc235-df5d-4c01-ad13-af627e3f9146_S-2306-012864_HE_2of2.svs region 1
# img = readsvs(svs_path, 0, (20000,19000), (3000, 3000)) # 7c542fa8-07ce-40d4-a993-be47180302f9_S-2006-003965_HE_2of2.svs region 0
# img = readsvs(svs_path, 0, (22000,23500), (3000, 3000)) # 7c542fa8-07ce-40d4-a993-be47180302f9_S-2006-003965_HE_2of2.svs region 1
# img = readsvs(svs_path, 0, (25000,24500), (3000, 3000)) # 7c542fa8-07ce-40d4-a993-be47180302f9_S-2006-003965_HE_2of2.svs region 2
# img = readsvs(svs_path, 0, (77250,25000), (3000, 3000)) # 32bacae4-3217-4443-bc7c-467b2eb2e347_S-2107-015040_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (73750,25000), (3000, 3000)) # 32bacae4-3217-4443-bc7c-467b2eb2e347_S-2107-015040_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (68500,27000), (3000, 3000)) # 32bacae4-3217-4443-bc7c-467b2eb2e347_S-2107-015040_TRI_1of2.svs region 2
# img = readsvs(svs_path, 0, (62500,29000), (3000, 3000)) # 32bacae4-3217-4443-bc7c-467b2eb2e347_S-2107-015040_TRI_1of2.svs region 3
# img = readsvs(svs_path, 0, (8500, 13500), (3000, 3000)) # a70271b6-70b4-404c-8e00-405b8acb56b1_S-1910-000139_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (8500, 15500), (3000, 3000)) # a70271b6-70b4-404c-8e00-405b8acb56b1_S-1910-000139_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (40000, 35500), (3000, 3000)) # 571cd693-92b4-4a59-94af-444864967d2c_S-2407-010708_PAS_2of2.svs region 0
# img = readsvs(svs_path, 0, (41000, 40500), (3000, 3000)) # 571cd693-92b4-4a59-94af-444864967d2c_S-2407-010708_PAS_2of2.svs region 1
# img = readsvs(svs_path, 0, (42000, 48500), (3000, 3000)) # 571cd693-92b4-4a59-94af-444864967d2c_S-2407-010708_PAS_2of2.svs region 2
# img = readsvs(svs_path, 0, (41000, 53500), (3000, 3000)) # 571cd693-92b4-4a59-94af-444864967d2c_S-2407-010708_PAS_2of2.svs region 3
# img = readsvs(svs_path, 0, (40500, 56750), (3000, 3000)) # 571cd693-92b4-4a59-94af-444864967d2c_S-2407-010708_PAS_2of2.svs region 4
# img = readsvs(svs_path, 0, (82250, 30500), (3000, 3000)) # 11edf527-6c6b-42a6-bf0f-d9b638679abd_S-2102-006599_PAS_1of2.svs region 0
# img = readsvs(svs_path, 0, (82250, 30500), (3000, 3000)) # 138ad41e1-b8ec-4ca4-a9ee-f969cd4820ba_S-2310-016644_HE_1of2.svs region 0
# img = readsvs(svs_path, 0, (15000, 3500), (3000, 3000)) # 97fe61fb-b97a-44e3-b4a6-258990851278_S-2209-009414_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (100500, 13000), (3000, 3000)) # e30e8020-2d64-4b44-b974-1ee7e4dd309c_S-2303-008882_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (89500, 22000), (3000, 3000)) # e30e8020-2d64-4b44-b974-1ee7e4dd309c_S-2303-008882_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (87500, 24000), (3000, 3000)) # e30e8020-2d64-4b44-b974-1ee7e4dd309c_S-2303-008882_TRI_1of2.svs region 2
# img = readsvs(svs_path, 0, (83500, 25000), (3000, 3000)) # e30e8020-2d64-4b44-b974-1ee7e4dd309c_S-2303-008882_TRI_1of2.svs region 3
# img = readsvs(svs_path, 0, (9750, 28000), (3000, 3000)) # a1f207cb-2b88-46dc-bcba-fdb70f8390c9_S-2105-008099_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (10000, 31500), (3000, 3000)) # a1f207cb-2b88-46dc-bcba-fdb70f8390c9_S-2105-008099_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (10000, 36500), (3000, 3000)) # a1f207cb-2b88-46dc-bcba-fdb70f8390c9_S-2105-008099_SIL_1of2.svs region 2
# img = readsvs(svs_path, 0, (14500, 18500), (3000, 3000)) # b704aaf2-1f3b-4ac0-adb6-d87cc1c78b43_S-2301-001224_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (13500, 23500), (3000, 3000)) # b704aaf2-1f3b-4ac0-adb6-d87cc1c78b43_S-2301-001224_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (16500, 6500), (3000, 3000)) # 91c2c40b-3b3d-474c-8476-b72fa9b35904_S-2308-002465_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (15500, 19500), (3000, 3000)) # 91c2c40b-3b3d-474c-8476-b72fa9b35904_S-2308-002465_TRI_2of2.svs region 1
# img = readsvs(svs_path, 0, (2500, 8000), (3000, 3000)) # c84035c0-4aec-4efe-ae99-2bbbad9e3390_S-2303-014334_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (5000, 12250), (3000, 3000)) # c84035c0-4aec-4efe-ae99-2bbbad9e3390_S-2303-014334_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (9500, 19250), (3000, 3000)) # c84035c0-4aec-4efe-ae99-2bbbad9e3390_S-2303-014334_TRI_1of2.svs region 2
# img = readsvs(svs_path, 0, (13000, 28000), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 0
# img = readsvs(svs_path, 0, (18000, 24000), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 1
# img = readsvs(svs_path, 0, (24000, 17000), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 2
# img = readsvs(svs_path, 0, (26000, 15500), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 3
# img = readsvs(svs_path, 0, (14000, 21500), (3000, 3000)) # 58a620f1-0702-42b7-bfb7-24a04a40c864_S-2310-016662_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (22500, 16000), (3000, 3000)) # 655bc49d-8b84-4493-a4c2-1407acc52bd0_S-2006-001850_SIL_2of2.svs region 0
# img = readsvs(svs_path, 0, (25500, 16000), (3000, 3000)) # 655bc49d-8eb84-4493-a4c2-1407acc52bd0_S-2006-001850_SIL_2of2.svs region 1
# img = readsvs(svs_path, 0, (6500, 23500), (3000, 3000)) # 7115f02d-a65b-42b9-bcc0-39a64b0fc496_S-2402-002768_PAS_1of2.svs region 0
# img = readsvs(svs_path, 0, (7000, 25500), (3000, 3000)) # 7115f02d-a65b-42b9-bcc0-39a64b0fc496_S-2402-002768_PAS_1of2.svs region 1
# img = readsvs(svs_path, 0, (9000, 22500), (3000, 3000)) # 7115f02d-a65b-42b9-bcc0-39a64b0fc496_S-2402-002768_PAS_1of2.svs region 2
# img = readsvs(svs_path, 0, (59000, 18000), (3000, 3000)) # 4408f62c-b33d-4579-97db-15e8e087f25c_S-2203-016181_SIL_1of1.svs region 0
# img = readsvs(svs_path, 0, (148000, 4500), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (139500, 15500), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (132500, 20000), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 2
# img = readsvs(svs_path, 0, (129500, 7500), (3000, 3000)) # 5a91767b-cff9-423f-afb3-0f8cf256d20c_S-2402-002819_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (9500, 44000), (3000, 3000)) # c251cc03-b1ed-4688-b30d-253dae319f7d_S-2006-004860_SIL_2of2.svs region 0
# img = readsvs(svs_path, 0, (12500, 11250), (3000, 3000)) # c251cc03-b1ed-4688-b30d-253dae319f7d_S-2006-004860_SIL_2of2.svs region 1
# img = readsvs(svs_path, 0, (7500, 8000), (3000, 3000)) # cec0e7b7-d993-47f6-9754-e1db0bf50e15_S-2305-006489_HE_1of2.svs region 0
# img = readsvs(svs_path, 0, (51250, 43750), (3000, 3000)) # ea4f5e11-7f89-4e40-99d3-bd97c630d83d_S-1910-000094_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (54250, 39750), (3000, 3000)) # ea4f5e11-7f89-4e40-99d3-bd97c630d83d_S-1910-000094_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (59250, 36750), (3000, 3000)) # ea4f5e11-7f89-4e40-99d3-bd97c630d83d_S-1910-000094_TRI_1of2.svs region 2
# img = readsvs(svs_path, 0, (64250, 27750), (3000, 3000)) # ea4f5e11-7f89-4e40-99d3-bd97c630d83d_S-1910-000094_TRI_1of2.svs region 3

# region selection for vessel training
# img = readsvs(svs_path, 0, (142000, 12000), (3000, 3000)) # 76d12d87-6bef-457d-a568-5241e5db4be9_S-2305-012743_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (145000, 10000), (3000, 3000)) # 76d12d87-6bef-457d-a568-5241e5db4be9_S-2305-012743_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (11000, 24500), (3000, 3000)) # 255e6f9d-8cc5-4818-9dca-fe28c3307d57_S-1905-017784_PAS_2of2.svs region 0
# img = readsvs(svs_path, 0, (10500, 29500), (3000, 3000)) # 255e6f9d-8cc5-4818-9dca-fe28c3307d57_S-1905-017784_PAS_2of2.svs region 1
# img = readsvs(svs_path, 0, (24000, 14000), (3000, 3000)) # bde444a9-f51b-4baf-95e4-ce3bb350809b_S-2409-010153_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (27000, 14000), (3000, 3000)) # bde444a9-f51b-4baf-95e4-ce3bb350809b_S-2409-010153_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (29000, 16000), (3000, 3000)) # bde444a9-f51b-4baf-95e4-ce3bb350809b_S-2409-010153_TRI_1of2.svs region 1
# img = readsvs(svs_path, 0, (9000, 10000), (3000, 3000)) # c59b275a-6eb7-446f-86a7-0667880d137a_S-2102-003503_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (7000, 12000), (3000, 3000)) # e0ecbd08-bad7-4256-a3e4-a13f0d91c408_S-2305-001374_HE_A1of2.svs region 0
# img = readsvs(svs_path, 0, (17000, 16000), (3000, 3000)) # 055c2d01-c256-4d42-a623-e3cc53603ceb_S-2311-007110_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (45500, 27500), (3000, 3000)) # 4408f62c-b33d-4579-97db-15e8e087f25c_S-2203-016181_SIL_1of1.svs region 0
# img = readsvs(svs_path, 0, (68500, 13500), (3000, 3000)) # 9348b44b-1bd0-47e5-9086-003f35a036c5_S-2311-011845_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (10000, 17000), (3000, 3000)) # 80313b48-7516-4fe4-8bc3-880212c0e6ab_S-2109-019202_TRI_1of2.svs region 0
# img = readsvs(svs_path, 0, (76000, 33000), (3000, 3000)) # a3d7c720-51e1-4c9f-a528-7bc6b13cd3e4_S-1905-017550_PAS_1of2.svs region 0
# img = readsvs(svs_path, 0, (81000, 30000), (3000, 3000)) # a3d7c720-51e1-4c9f-a528-7bc6b13cd3e4_S-1905-017550_PAS_1of2.svs region 1
# img = readsvs(svs_path, 0, (12000, 26500), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 4
# img = readsvs(svs_path, 0, (21500, 22000), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 5
# img = readsvs(svs_path, 0, (21500, 22000), (3000, 3000)) # fa64da26-b859-4cf5-9f36-615087edf1d1_S-2002-007541_HE_2of2.svs region 6
# img = readsvs(svs_path, 0, (2000, 4000), (3000, 3000)) # a5a99599-b25f-433c-ad45-3278f2481889_S-2106-003450_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (5000, 8000), (3000, 3000)) # 2d2bfed9-0ae2-4f66-bcc8-7f86363b58c3_S-1908-009782_PAS_1of2.svs region 0
# img = readsvs(svs_path, 0, (5000, 12000), (3000, 3000)) # 2d2bfed9-0ae2-4f66-bcc8-7f86363b58c3_S-1908-009782_PAS_1of2.svs region 1
# img = readsvs(svs_path, 0, (21500, 10000), (3000, 3000)) # 14b76a89-7820-4934-b76d-808d4d75de2a_S-2311-013462_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (24000, 7000), (3000, 3000)) # 14b76a89-7820-4934-b76d-808d4d75de2a_S-2311-013462_TRI_2of2.svs region 1
# img = readsvs(svs_path, 0, (91000, 58000), (3000, 3000)) # 39dd4918-2283-463c-b7dd-8829b3572504_S-2302-010879_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (90750, 71000), (3000, 3000)) # 39dd4918-2283-463c-b7dd-8829b3572504_S-2302-010879_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (14000, 21000), (3000, 3000)) # 70b9748d-0a5a-4d47-b5b1-f59a24d9f2d4_S-2010-013045_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (16000, 21000), (3000, 3000)) # 70b9748d-0a5a-4d47-b5b1-f59a24d9f2d4_S-2010-013045_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (148500, 6000), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 3
# img = readsvs(svs_path, 0, (145500, 11000), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 4
# img = readsvs(svs_path, 0, (135500, 17500), (3000, 3000)) # 256b8d9b-47b9-43cc-a3bb-d47a4b83bacc_S-2404-003772_TRI_1of2.svs region 5
# img = readsvs(svs_path, 0, (10000, 11000), (3000, 3000)) # 712eb63f-d6b4-4d7f-a84a-60a6964648a0_S-2305-006507_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (10500, 13500), (3000, 3000)) # 712eb63f-d6b4-4d7f-a84a-60a6964648a0_S-2305-006507_TRI_2of2.svs region 1
# img = readsvs(svs_path, 0, (13500, 19500), (3000, 3000)) # 712eb63f-d6b4-4d7f-a84a-60a6964648a0_S-2305-006507_TRI_2of2.svs region 2
# img = readsvs(svs_path, 0, (4000, 2000), (3000, 3000)) # 5605b2fa-570d-4d87-b0c8-97cd8fbb254e_S-2306-012910_HE_1of2.svs region 0
# img = readsvs(svs_path, 0, (4000, 13000), (3000, 3000)) # 5605b2fa-570d-4d87-b0c8-97cd8fbb254e_S-2306-012910_HE_1of2.svs region 1
# img = readsvs(svs_path, 0, (106000, 26500), (3000, 3000)) # 22302e71-6d12-46b9-aace-c2a3e0eac12d_S-2308-002464_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (108000, 23500), (3000, 3000)) # 22302e71-6d12-46b9-aace-c2a3e0eac12d_S-2308-002464_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (8000, 11000), (3000, 3000)) # 33570ef5-72f0-4c94-832e-6a3261499b04_S-2109-019248_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (9000, 14000), (3000, 3000)) # 33570ef5-72f0-4c94-832e-6a3261499b04_S-2109-019248_TRI_2of2.svs region 1
# img = readsvs(svs_path, 0, (12000, 17000), (3000, 3000)) # 33570ef5-72f0-4c94-832e-6a3261499b04_S-2109-019248_TRI_2of2.svs region 2
# img = readsvs(svs_path, 0, (18000, 20000), (3000, 3000)) # 55417dab-a4cd-47d5-a2c0-474d4435dc9c_S-1908-010163_TRI_2of2.svs region 0
# img = readsvs(svs_path, 0, (20250, 18000), (3000, 3000)) # 55417dab-a4cd-47d5-a2c0-474d4435dc9c_S-1908-010163_TRI_2of2.svs region 1
# img = readsvs(svs_path, 0, (22500, 20000), (3000, 3000)) # 58162d0e-8531-45c2-86de-8d7d6a228905_S-2308-017077_HE_2of2.svs region 0
# img = readsvs(svs_path, 0, (10500, 24000), (3000, 3000)) # c251cc03-b1ed-4688-b30d-253dae319f7d_S-2006-004860_SIL_2of2.svs region 2
# img = readsvs(svs_path, 0, (6000, 32500), (3000, 3000)) # cbfcff78-62e9-412c-8bc0-763cdf880187_S-2102-006716_PAS_2of2.svs region 0
# img = readsvs(svs_path, 0, (9000, 31000), (3000, 3000)) # cbfcff78-62e9-412c-8bc0-763cdf880187_S-2102-006716_PAS_2of2.svs region 1
# img = readsvs(svs_path, 0, (21000, 22000), (3000, 3000)) # cbfcff78-62e9-412c-8bc0-763cdf880187_S-2102-006716_PAS_2of2.svs region 2
# img = readsvs(svs_path, 0, (8000, 30000), (3000, 3000)) # d0cea8b8-ca12-4781-834c-ec5a9e5b6008_S-2010-004217_HE_2of2.svs region 0
# img = readsvs(svs_path, 0, (6500, 23000), (3000, 3000)) # d0cea8b8-ca12-4781-834c-ec5a9e5b6008_S-2010-004217_HE_2of2.svs region 1
# img = readsvs(svs_path, 0, (6500, 15000), (3000, 3000)) # d0cea8b8-ca12-4781-834c-ec5a9e5b6008_S-2010-004217_HE_2of2.svs region 2
# img = readsvs(svs_path, 0, (73000, 37000), (3000, 3000)) # 6e77144d-bb51-4847-b003-a90b53a27832_S-2107-014944_SIL_1of2.svs region 0
# img = readsvs(svs_path, 0, (60000, 46500), (3000, 3000)) # 6e77144d-bb51-4847-b003-a90b53a27832_S-2107-014944_SIL_1of2.svs region 1
# img = readsvs(svs_path, 0, (21000, 9000), (3000, 3000)) # 096a0096-4dec-47c9-8e07-b5a508c24206_S-2302-002668_HE_1of2.svs region 0
# img = readsvs(svs_path, 0, (11500, 18000), (3000, 3000)) # 6395e880-aa4a-4a31-ac1e-140ae6421807_S-2106-003495_SIL_2of2.svs region 0
# img = readsvs(svs_path, 0, (11500, 12000), (3000, 3000)) # 351240e1-3907-4f9a-b53b-cd4b9eebb37e_S-2108-000691_SIL_2of2.svs region 0
# img = readsvs(svs_path, 0, (12500, 21000), (3000, 3000)) # 351240e1-3907-4f9a-b53b-cd4b9eebb37e_S-2108-000691_SIL_2of2.svs region 1
# img = readsvs(svs_path, 0, (27000, 12000), (3000, 3000)) # daea1032-d81b-45d3-8d26-831c141d2cab_S-2010-012855_PAS_1of2.svs region 0
img = readsvs(svs_path, 0, (19000, 30000), (3000, 3000)) # daea1032-d81b-45d3-8d26-831c141d2cab_S-2010-012855_PAS_1of2.svs region 1

# cluster regions
# img = readsvs(svs_path, 0, (39500, 23000), (3000, 3000)) # BR21-2049-B-A-1-9-TRICHROME - 2022-11-09 16.43.38.ndpi region slide 54
# img = readsvs(svs_path, 0, (41500, 21000), (3000, 3000)) # BR21-2049-B-A-1-9-TRICHROME - 2022-11-09 16.43.38.ndpi region slide 53
# img = readsvs(svs_path, 0, (23500, 33500), (3000, 3000)) # BR21-2049-B-A-1-9-TRICHROME - 2022-11-09 16.43.38.ndpi region slide 52
# img = readsvs(svs_path, 0, (12000, 26500), (3000, 3000)) # BR22-2090-A-1-9-TRICHROME - 2022-11-11 16.40.02.ndpi region 0
# img = readsvs(svs_path, 0, (45000, 11500), (3000, 3000)) # 21-2007 A-1-9 Trich - 2021-03-22 13.34.30.ndpi region 0
# img = readsvs(svs_path, 0, (54000, 13000), (3000, 3000)) # 21-2007 A-1-9 Trich - 2021-03-22 13.34.30.ndpi region 1
# img = readsvs(svs_path, 0, (60000, 35500), (3000, 3000)) # 21-2007 A-1-9 Trich - 2021-03-22 13.34.30.ndpi region 2
# img = readsvs(svs_path, 0, (14000, 22500), (3000, 3000)) # BR22-2093-A-1-9-TRICHROME - 2022-11-11 17.12.35.ndpi region 0
# img = readsvs(svs_path, 0, (14000, 31500), (3000, 3000)) # BR22-2093-A-1-9-TRICHROME - 2022-11-11 17.12.35.ndpi region 1
# img = readsvs(svs_path, 0, (8500, 22500), (3000, 3000)) # 21-2015 A-1-9 Trich - 2021-03-22 16.35.02.ndpi region 0
# img = readsvs(svs_path, 0, (5500, 37000), (3000, 3000)) # BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18.ndpi region 0
# img = readsvs(svs_path, 0, (40500, 34000), (3000, 3000)) # BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18.ndpi region 1
# img = readsvs(svs_path, 0, (14500, 20000), (3000, 3000)) # 896801 - 2021-07-23 11.13.14.ndpi region 0
# img = readsvs(svs_path, 0, (14000, 19000), (2000, 2000)) # 896801 - 2021-07-23 11.13.14.ndpi region 1
# img = readsvs(svs_path, 0, (24000, 7000), (3000, 3000)) # BR22-2073-A-1-9-TRI - 2022-08-08 15.03.42.ndpi region 0
# img = readsvs(svs_path, 0, (27000, 31000), (3000, 3000)) # BR21-2049-B-A-1-9-TRICHROME - 2022-11-09 16.43.38.ndpi region 0

# larger cluster regions
# img = readsvs(svs_path, 0, (20000, 30000), (10000, 10000)) # BR21-2049-B-A-1-9-TRICHROME - 2022-11-09 16.43.38.ndpi region slide 52 big
# img = readsvs(svs_path, 0, (60000, 34500), (10000, 8000)) # 21-2007 A-1-9 Trich - 2021-03-22 13.34.30.ndpi region 2 big
# img = readsvs(svs_path, 0, (42000, 10500), (10000, 8000)) # 21-2007 A-1-9 Trich - 2021-03-22 13.34.30.ndpi region 0 big
# img = readsvs(svs_path, 0, (7500, 20500), (10000, 8000)) # 21-2015 A-1-9 Trich - 2021-03-22 16.35.02.ndpi region 0 big 
# img = readsvs(svs_path, 0, (14000, 18500), (6000, 10000)) # BR22-2093-A-1-9-TRICHROME - 2022-11-11 17.12.35.ndpi region 0
# img = readsvs(svs_path, 0, (17000, 5000), (16000, 8000)) # BR22-2073-A-1-9-TRI - 2022-08-08 15.03.42.ndpi region 0
# img = readsvs(svs_path, 0, (3000, 35000), (10000, 8000)) # BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18.ndpi region 0
# img = readsvs(svs_path, 0, (40500, 32000), (10000, 8000)) # BR21-2023-B-A-1-9-TRICHROME - 2022-11-09 14.48.18.ndpi region 1

plt.imshow(img)
plt.show()

img.save(svs_folder / (svs_name + f"_{identifier}" + ".tiff"))