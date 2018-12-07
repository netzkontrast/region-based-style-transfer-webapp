<template>
  <div class="RegionBasedStyleTransfer">
    <div>
      <div class="container">
        <h4><span class="badge badge-primary">Sample Images</span></h4>
        <div class="row">
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample1')" :src="sampleImages.sample1" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample2')" :src="sampleImages.sample2" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample3')" :src="sampleImages.sample3" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample4')" :src="sampleImages.sample4" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample5')" :src="sampleImages.sample5" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top sampleImage" style="cursor:pointer" @click="chooseSample('sample6')" :src="sampleImages.sample6" alt="Card image cap">
          </div>
        </div>

        <h4 class="mt-1"><span class="badge badge-primary">Result</span></h4>
        <div class="row">
          <div class="col-sm-3 resultRow">
            <div class="card">
              <img class="card-img-top showResult" :src="originImage" @click="$refs.fileInput.click()" style="cursor:pointer"  alt="Card image cap">
              <input id="input-image" style="display:none;" type="file" @change="onFileSelected" ref="fileInput" accept=".jpg, .jpeg">
              <div class="card-body" style="height:42px; position:relative;text-align:center;">
                <div class="my-3">
                  <h5 style="display:inline; bottom:0px;"><span class="badge badge-info">Origin</span></h5>
                </div>
              </div>
            </div>
          </div>

          <div class="col-sm-3 resultRow">
            <div class="card">
              <img class="card-img-top showResult" :src="globalStyleTransferImage" alt="Card image cap">
              <div class="card-body" style="height:42px; position:relative;text-align:center;">
                <div class="my-3">
                  <h5 style="display:inline; bottom:0px;"><span class="badge badge-info">Global Style Transfer</span></h5>
                </div>
              </div>
            </div>
          </div>

          <div class="col-sm-3 resultRow">
            <div class="card">
              <img class="card-img-top showResult" :src="regionBasedStyleTransferImage" alt="Card image cap">
              <div class="card-body" style="height:42px; position:relative;text-align:center;">
                <div class="my-3">
                  <h5 style="display:inline; bottom:0px;"><span class="badge badge-info">Background Style Transfer</span></h5>
                </div>
              </div>
            </div>
          </div>

          <div class="col-sm-3 resultRow">
            <div class="card">
              <img class="card-img-top showResult" :src="regionBasedStyleTransferWithColorImage" alt="Card image cap">
              <div class="card-body" style="height:42px; position:relative;text-align:center;">
                <div class="my-3">
                  <h5 style="display:inline; bottom:0px;"><span class="badge badge-info">Background Style Transfer (add color)</span></h5>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <h4 class="mt-1"><span class="badge badge-primary">Styles</span></h4>
        <div class="row">
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style1')" :src="styleImages.style1" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style2')" :src="styleImages.style2" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style3')" :src="styleImages.style3" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style4')" :src="styleImages.style4" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style5')" :src="styleImages.style5" alt="Card image cap">
          </div>
          <div class="card col-sm-2">
            <img class="card-img-top style" style="cursor:pointer" @click="applyStyle('style6')" :src="styleImages.style6" alt="Card image cap">
          </div>
        </div>
      </div>
    </div>
    <modal id="svgModal" name="loadingModal" width="300px" height="350px" :clickToClose="false">
      <img :src="svgPath" alt="loading" style="vertical-align: middle;" width="300px" height="300px"/>
      <h4  style="text-align:center;" ><span class="badge badge-warning">Style Transfering ...</span></h4>
    </modal>

    <v-dialog id="alertModel" name="alertModal" width="300px" height="80px">
    </v-dialog>
    
  </div>
</template>

<script>
import axios from 'axios';

const STYLE_TYPES = {
  "style1": "acrylic-1143758",
  "style2": "chris-barbalis",
  "style3": "dt3108",
  "style4": "scream",
  "style5": "udnie",
  "style6": "dt1966",
};
const SAMPLE_IMAGES = {
  "SAMPLE_IMAGE1": "girl1.jpg",
  "SAMPLE_IMAGE2": "girl2.jpg",
  "SAMPLE_IMAGE3": "VOC2010_18.jpg",
  "SAMPLE_IMAGE5": "hong_ps.jpg",
  "SAMPLE_IMAGE4": "man2.jpg",
  "SAMPLE_IMAGE6": "man1.jpg"
};
const STYLE_IMAGES = {
  "STYLE_IMAGE1": STYLE_TYPES.style1 + ".jpg",
  "STYLE_IMAGE2": STYLE_TYPES.style2 + ".jpg",
  "STYLE_IMAGE3": STYLE_TYPES.style3 + ".jpg",
  "STYLE_IMAGE4": STYLE_TYPES.style4 + ".jpg",
  "STYLE_IMAGE5": STYLE_TYPES.style5 + ".jpg",
  "STYLE_IMAGE6": STYLE_TYPES.style6 + ".jpg"
};
const PLACEHOLDER_IMAGE = "https://via.placeholder.com/378x270.png";
const UPLOAD_PLACEHOLDER_IMAEG = "https://placehold.it/100/000000/ffffff?text=UPLOAD";

export default {
  name: 'RegionBasedStyleTransfer',
  data () {
    return {
      selectedFile: undefined,
      originImage: UPLOAD_PLACEHOLDER_IMAEG,
      globalStyleTransferImage: PLACEHOLDER_IMAGE,
      regionBasedStyleTransferImage: PLACEHOLDER_IMAGE,
      regionBasedStyleTransferWithColorImage: PLACEHOLDER_IMAGE,
      sampleImages: {
        "sample1": "https://via.placeholder.com/150",
        "sample2": "https://via.placeholder.com/150",
        "sample3": "https://via.placeholder.com/150",
        "sample4": "https://via.placeholder.com/150",
        "sample5": "https://via.placeholder.com/150",
        "sample6": "https://via.placeholder.com/150",
      },
      styleImages: {
        "style1": "https://via.placeholder.com/150",
        "style2": "https://via.placeholder.com/150",
        "style3": "https://via.placeholder.com/150",
        "style4": "https://via.placeholder.com/150",
        "style5": "https://via.placeholder.com/150",
        "style6": "https://via.placeholder.com/150",
      },
      demoImage: undefined,
      currentStyle: undefined,
      host: "localhost",
      port: "5000",
      //host: "35.245.30.254",
      //port: "5000",
      svgPath: undefined,
    }
  },
  methods: {
    loadingModalShow () {
      this.$modal.show('loadingModal');
    },
    loadingModalHide () {
      this.$modal.hide('loadingModal');
    },
    alertModalShow (alertReason, advice="") {
      this.$modal.show('dialog', {
        title: 'Error!',
        text: 'Error: ' + alertReason + " \n " + advice,
        buttons: [
          {
            title: 'Close'
          }
      ]
      })
    },
    updateSampleImage(){
      this.sampleImages.sample1 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE1;
      this.sampleImages.sample2 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE2;
      this.sampleImages.sample3 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE3;
      this.sampleImages.sample4 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE4;
      this.sampleImages.sample5 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE5;
      this.sampleImages.sample6 = "http://" + this.host + ":" + this.port + "/get_sample_image/" + SAMPLE_IMAGES.SAMPLE_IMAGE6;
    },
    updateStyleImage(){
      this.styleImages.style1 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE1;
      this.styleImages.style2 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE2;
      this.styleImages.style3 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE3;
      this.styleImages.style4 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE4;
      this.styleImages.style5 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE5;
      this.styleImages.style6 = "http://" + this.host + ":" + this.port + "/get_style_image/" + STYLE_IMAGES.STYLE_IMAGE6;
    },
    onMounted(){
      this.svgPath = "./static/Spinner-1s-200px.svg";
      this.updateSampleImage();
      this.updateStyleImage();
    },

    onFileSelected(event){
      this.selectedFile = event.target.files[0];
      this.demoImage = null;
      this.globalStyleTransferImage = PLACEHOLDER_IMAGE;
      this.regionBasedStyleTransferImage = PLACEHOLDER_IMAGE;
      this.regionBasedStyleTransferWithColorImage = PLACEHOLDER_IMAGE;

      if(this.selectedFile.size > 1048576){
        this.alertModalShow("File is too large!", "Please make sure the size of image is smaller than 1MB");
        console.log("File is too large! Please make sure the size of image is smaller than 1MB");
        this.selectedFile = null;
        document.querySelector('#input-image').value='';
        return false;
      }
      else if(this.selectedFile.type != "image/jpeg"){
        this.alertModalShow("Unsupport file type!", "Current only support jpg and jepg image file");
        console.log("Unsupport file type! Current only support jpg and jepg image file");
        this.selectedFile = null;
        document.querySelector('#input-image').value='';
        return false;;
      }
      else{
        document.querySelector('#input-image').value='';
        this.fileUpload();
      }
    },

    fileUpload(){
      if(this.selectedFile){
        var formData = new FormData();
        formData.append('image', this.selectedFile, this.selectedFile.name);
        axios.post("http://" + this.host + ":" + this.port + "/upload", formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: uploadEvent =>{
              console.log("Upload Progress: " + Math.round(uploadEvent.loaded / uploadEvent.total * 100));
            },
          })
        .then(res=>{
          console.log(res);
          this.originImage = "http://" + this.host + ":" + this.port + "/get_upload_image/" + this.selectedFile.name;
        });
      }
    },

    onUpload(style_type=STYLE_TYPES.style1){
      if(this.selectedFile){
        this.loadingModalShow();
        var formData = new FormData();
        formData.append('image', this.selectedFile, this.selectedFile.name);
        axios.post("http://" + this.host + ":" + this.port + "/"+style_type, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: uploadEvent =>{
              console.log("Upload Progress: " + Math.round(uploadEvent.loaded / uploadEvent.total * 100));
            },
          })
        .then(res=>{
          console.log(res);
          this.updateBlendImage(this.selectedFile.name, style_type);
        });
      }
      else if(this.demoImage){
        this.loadingModalShow();
        var image_name = this.demoImage.substring(this.demoImage.lastIndexOf("/")+1, this.demoImage.length);
        this.updateBlendImage(image_name, style_type);
      }
      else{
        this.alertModalShow("Please either use sample image or upload your own image!");
      }
    },

    updateBlendImage(image_name, style_type){
      console.log("func: updateBlendImage ----------");
      console.log("image_name: ", image_name);
      console.log("style_type: ", style_type);

      var index_dot = image_name.lastIndexOf(".");
      var filename = image_name.substring(0, index_dot);
      var suffix = image_name.substring(index_dot+1, image_name.length);

      this.globalStyleTransferImage = "http://" + this.host + ":" + this.port + "/get_global_style_transfer_image/" + filename + "_" + style_type + "." + suffix;
      this.regionBasedStyleTransferImage = "http://" + this.host + ":" + this.port + "/get_region_based_style_transfer_image/" + "blend_" + filename + "_" + style_type + "." + suffix;
      this.regionBasedStyleTransferWithColorImage = "http://" + this.host + ":" + this.port + "/get_region_based_style_transfer_with_color_image/" + "blend_" + filename + "_" + style_type + "_color" + "." + suffix;
      this.originImage = "http://" + this.host + ":" + this.port + "/get_upload_image/" + image_name;

      this.loadingModalHide();
    },

    chooseSample(sample_id){
      this.demoImage = this.sampleImages[sample_id];
      this.originImage = this.sampleImages[sample_id];
      this.selectedFile = null;
      this.globalStyleTransferImage = PLACEHOLDER_IMAGE;
      this.regionBasedStyleTransferImage = PLACEHOLDER_IMAGE;
      this.regionBasedStyleTransferWithColorImage = PLACEHOLDER_IMAGE;
    },

    applyStyle(stypeIdx){
      this.currentStyle = STYLE_TYPES[stypeIdx];
      console.log("style_type: ", this.currentStyle);
      this.onUpload(this.currentStyle);
    },
  },
  mounted: function () {
    this.onMounted();
  },
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.card-img-top {
  width: 100%;
  height: 15vw;
  object-fit: scale-down;
}
.card{
  padding-right: 0px;
  padding-left: 0px;
}

.card-body {
    padding:0.1rem;
}
.style{
  object-fit: cover;
}

.showResult{
  height:270px;
  object-fit:scale-down;
}

.sampleImage{
  height:100px;
}

.resultRow{
  padding-right: 0px;
  padding-left: 0px;
}

#svgModal{
   opacity: 0.9;
   line-height: 300px;
}
</style>
