#read h5 file and convert them to image and txt
import h5py
import os, cv2
root = "/home1/caixin/GazeData/VIPLGaze538/test"

out_root = "/data1/GazeData/vipl538/test4adapt"
scale = True
listdirs = lambda x: [f for f in os.listdir(x) if not f.startswith('.') and f.endswith('h5')] 
def ImageProcessing_MPII():
    persons = listdirs(root)
    persons.sort()
    for person in persons:
        im_root = os.path.join(root, person)
        print(im_root)
        person = person.split(".")[0]
        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "Label", "%s.label"%person)
        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))

        # print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, im_outpath, label_outpath, person):
    outfile = open(label_outpath, 'w')
    outfile.write("Face 2DGaze 2DHead \n")
    # if not os.path.exists(os.path.join(im_outpath, "face")):
    if True:
        with h5py.File(im_root, 'r') as f:
            for cam in f.keys():
                for video in f[cam].keys():
                    print(f"Start Processing {person} {cam} {video}")
                    sub_f = f[cam][video]
                    total = len(sub_f["face_patch"])
                    os.makedirs(os.path.join(im_outpath, "face",cam, video), exist_ok=True)

                    for i in range(total):
                        im_name = "%05d.png"%i
                        im_face  = sub_f["face_patch"][i, :]
                        gaze = sub_f["face_gaze"][i, :2]
                        gaze = ",".join(gaze.astype("str"))
                        im_path = os.path.join(im_outpath, "face", cam, video, im_name)
                        #cover to BGR
                        im_face = cv2.cvtColor(im_face, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(im_path, im_face)
                        save_name_face = os.path.join(person, "face", cam, video, im_name)
                        save_str = " ".join([save_name_face, gaze, gaze])
                        outfile.write(save_str + "\n")
        print("")
    outfile.close()
    # Image Processing 
 

if __name__ == "__main__":
    ImageProcessing_MPII()