#include <NeuralNetwork/Blobs/OriginalSample.h>


namespace NN {



void OriginalSample::Pad(const int nsize, bool random_shift) {

    int nlength=nsize*nsize;
        std::vector<unsigned char> res(nlength,255);

        avir :: CImageResizer<> ImageResizer( 8 );
        if (this->matrix_cols>nsize||this->matrix_rows>nsize) {
            unsigned char *resize_result;
            int new_width;
            int new_height;

            if (this->matrix_cols>this->matrix_rows) {
                new_height=((float)nsize/(float)this->matrix_cols)*this->matrix_rows;
                if (new_height==0)  new_height=1;/*LOG << "failure\n width/height " << matrix_cols << "/" << matrix_rows << "\n";*/
                new_width=nsize;
            }
            else {
                new_width=((float)nsize/(float)this->matrix_rows)*this->matrix_cols;
                if (new_width==0) new_width=1;
                new_height=nsize;
            }
            resize_result=(unsigned char*)std::malloc(new_height*new_width*sizeof(unsigned char));
            ImageResizer.resizeImage( &data_matrix[0], matrix_cols, matrix_rows, 0, resize_result,new_width, new_height, 1, 0 );
            this->data_matrix=std::vector<unsigned char>(resize_result,resize_result+new_width*new_height);
            this->matrix_cols=new_width;
            this->matrix_rows=new_height;
        }

        int vpadding=nsize-this->matrix_rows;
        int hpadding=nsize-this->matrix_cols;

        int toppadding=0; int lpadding=0;

        if (random_shift) {
            std::random_device dev;
            std::default_random_engine e1(dev());
            if (vpadding>0) {
                std::uniform_int_distribution<int> dist1(0,vpadding);
                toppadding=dist1(e1);
            }
            if (hpadding>0) {
            std::uniform_int_distribution<int> dist2(0,hpadding);
            lpadding=dist2(e1);
            }
        }
        else {
            toppadding=vpadding-vpadding/2;
            lpadding=hpadding-hpadding/2;
        }

        for (int y=0;y<this->matrix_rows;y++) {
            int roff=(toppadding+y)*nsize;
            for (int x=0;x<this->matrix_cols; x++) {
                res[roff+lpadding+x]=this->data_matrix[y*this->matrix_cols+x];
            }
        }
        unsigned char ncols=nsize;
        unsigned char nrows=nsize;
        *this=OriginalSample(res,this->labels_short,ncols, nrows, this->labels.size());

}

void OriginalSample::resize(int nsize, bool preserve_aspect_ratio) {


    if (preserve_aspect_ratio) {
            int nlength=nsize*nsize;
            std::vector<unsigned char> res(nlength,255);

            avir :: CImageResizer<> ImageResizer( 8 );
            if (this->matrix_cols>nsize||this->matrix_rows>nsize) {
                unsigned char *resize_result;
                int new_width;
                int new_height;

                if (this->matrix_cols>this->matrix_rows) {
                    new_height=((float)nsize/(float)this->matrix_cols)*this->matrix_rows;
                    if (new_height==0)  new_height=1;
                    new_width=nsize;
                }
                else {
                    new_width=((float)nsize/(float)this->matrix_rows)*this->matrix_cols;
                    if (new_width==0) new_width=1;
                    new_height=nsize;
                }
                resize_result=(unsigned char*)std::malloc(new_height*new_width*sizeof(unsigned char));
                ImageResizer.resizeImage( &data_matrix[0], matrix_cols, matrix_rows, 0, resize_result,new_width, new_height, 1, 0 );
                this->data_matrix=std::vector<unsigned char>(resize_result,resize_result+new_width*new_height);
                this->matrix_cols=new_width;
                this->matrix_rows=new_height;
            }

            int vpadding=nsize-this->matrix_rows;
            int hpadding=nsize-this->matrix_cols;
            int toppadding=vpadding-vpadding/2;
            int lpadding=hpadding-hpadding/2;

            for (int y=0;y<this->matrix_rows;y++) {
                int roff=(toppadding+y)*nsize;
                for (int x=0;x<this->matrix_cols; x++) {
                    res[roff+lpadding+x]=this->data_matrix[y*this->matrix_cols+x];
                }
            }
            unsigned char ncols=nsize;
            unsigned char nrows=nsize;
            *this=OriginalSample(res,this->labels_short,ncols, nrows, this->labels.size());
        }
        else {
            int nlength=nsize*nsize;
            std::vector<unsigned char> res(nlength,255);
            avir :: CImageResizer<> ImageResizer( 8 );
            unsigned char *resize_result;
            int new_width=nsize;
            int new_height=nsize;
            resize_result=(unsigned char*)std::malloc(new_height*new_width*sizeof(unsigned char));
            ImageResizer.resizeImage( &data_matrix[0], matrix_cols, matrix_rows, 0, resize_result,new_width, new_height, 1, 0 );


            this->data_matrix=std::vector<unsigned char>(resize_result,resize_result+new_width*new_height);
            unsigned char ncols=nsize;
            unsigned char nrows=nsize;
            *this=OriginalSample(this->data_matrix,this->labels_short,ncols, nrows, this->labels.size());
        }
}


OriginalSample::OriginalSample(std::vector<uchar> original_uchar_data, uchar cols, uchar rows) {
    data_matrix=original_uchar_data;
    data_matrix_f.resize(data_matrix.size());
    for (int i=0;i<data_matrix.size();i++) {
        data_matrix_f[i]=1.0f-((float)data_matrix[i])/255.0f;
    }
    matrix_cols=cols;
    matrix_rows=rows;
}

OriginalSample::OriginalSample(std::vector<uchar> original_uchar_data, std::vector<uchar> lbls, uchar cols, uchar rows, int class_count) {
    data_matrix=original_uchar_data;
    data_matrix_f.resize(data_matrix.size());
    for (int i=0;i<data_matrix.size();i++) {
        data_matrix_f[i]=1.0f-((float)data_matrix[i])/255.0f;
    }
    labels_short=lbls;
    labels.resize(class_count,0.0f);
    for (auto lbl_class : labels_short) labels[lbl_class]=1.0f;

    matrix_cols=cols;
    matrix_rows=rows;
}

}
