<template>
  <div class="RegionBasedStyleTransfer">
    <div>
      <div class="container">
        <div class="row">
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample1" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample2" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample3" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample4" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample5" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="sampleImages.sample6" alt="Card image cap">
          </div>
        </div>
        </br>
        <div class="row">
          <div class="col-sm-4">
            <div class="card">
              <img class="card-img-top" :src="originImage" alt="Card image cap">
              <div class="card-body" style="height:42px">
                <input style="display:none" type="file" @change="onFileSelected" ref="fileInput">
                <button  style="float:left" class="btn btn-primary btn" @click="$refs.fileInput.click()"> Upload </button>

                <h3 style="display:inline"><span class="badge badge-info">Origin</span></h3>
                <button class="btn btn-primary btn" style="float:right" @click="onUpload"> Apply </button>
              </div>
            </div>
          </div>

          <div class="col-sm-4">
            <div class="card" >
              <img class="card-img-top" :src="globalStyleTransferImage" alt="Card image cap">
              <div class="card-body" style="height:42px">
                <h3><span class="badge badge-info">Global Style Transfer</span></h3>
              </div>
            </div>
          </div>

          <div class="col-sm-4">
            <div class="card">
              <img class="card-img-top" :src="regionBasedStyleTransferImage" alt="Card image cap">
              <div class="card-body" style="height:42px">
                <h3><span class="badge badge-info">Region-based Style Transfer</span></h3>
              </div>
            </div>
          </div>
        </div>
        </br>
        <div class="row">
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style1" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style2" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style3" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style4" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style5" alt="Card image cap">
          </div>
          <div class="card col-sm-2" style="width: 8rem;">
            <img class="card-img-top" :src="styleImages.style6" alt="Card image cap">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'RegionBasedStyleTransfer',
  data () {
    return {
      selectedFile: null,
      originImage: "https://via.placeholder.com/200",
      globalStyleTransferImage: "https://via.placeholder.com/200",
      regionBasedStyleTransferImage: "https://via.placeholder.com/200",
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
    }
  },
  methods: {
    updateSampleImage(){
      var SAMPLE_IMAGE_NAME1 = "TLPS-7103_small.jpg";
      var SAMPLE_IMAGE_NAME2 = "bird.jpg";
      var SAMPLE_IMAGE_NAME3 = "VOC2010_18.jpg";
      var SAMPLE_IMAGE_NAME4 = "rhino.jpg";
      var SAMPLE_IMAGE_NAME5 = "hong_ps.jpg";
      var SAMPLE_IMAGE_NAME6 = "image1.jpg";

      this.sampleImages.sample1 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME1;
      this.sampleImages.sample2 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME2;
      this.sampleImages.sample3 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME3;
      this.sampleImages.sample4 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME4;
      this.sampleImages.sample5 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME5;
      this.sampleImages.sample6 = "http://localhost:5000/get_sample_image/" + SAMPLE_IMAGE_NAME6;
    },

    updateStyleImage(){
      var STYLE_IMAGE_NAME1 = "la_muse.jpg";
      var STYLE_IMAGE_NAME2 = "rain_princess.jpg";
      var STYLE_IMAGE_NAME3 = "the_scream.jpg";
      var STYLE_IMAGE_NAME4 = "the_shipwreck_of_the_minotaur.jpg";
      var STYLE_IMAGE_NAME5 = "udnie.jpg";
      var STYLE_IMAGE_NAME6 = "wave.jpg";

      this.styleImages.style1 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME1;
      this.styleImages.style2 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME2;
      this.styleImages.style3 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME3;
      this.styleImages.style4 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME4;
      this.styleImages.style5 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME5;
      this.styleImages.style6 = "http://localhost:5000/get_style_image/" + STYLE_IMAGE_NAME6;
    },
    
    onMounted(){
      this.updateSampleImage();
      this.updateStyleImage();
    },

    onFileSelected(event){
      this.selectedFile = event.target.files[0];
      this.onUpload();
      this.originImage = "http://localhost:5000/get_upload_image/" + this.selectedFile.name;
    },
    onUpload(){
      var formData = new FormData();
      formData.append('image', this.selectedFile, this.selectedFile.name);
      axios.post("http://localhost:5000", formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: uploadEvent =>{
            console.log("Upload Progress: " + Math.round(uploadEvent.loaded / uploadEvent.total * 100));
          },
        })
      .then(res=>{
        console.log(res);
        this.updateBlendImage();
      });
    },
    updateBlendImage(){
      console.log("updateBlendImage");
      var index_dot = this.selectedFile.name.lastIndexOf(".");
      var filename = this.selectedFile.name.substring(0, index_dot);
      var suffix = this.selectedFile.name.substring(index_dot+1, this.selectedFile.name.length);

      this.globalStyleTransferImage = "http://localhost:5000/get_global_style_transfer_image/" + filename + "_wreck" + "." + suffix;
      this.regionBasedStyleTransferImage = "http://localhost:5000/get_region_based_style_transfer_image/" + "blend_" + filename + "_wreck" + "." + suffix;
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
    object-fit: cover;
}
.card-body {
    padding:0.1rem;
}
</style>
