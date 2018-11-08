<template>
  <div class="RegionBasedStyleTransfer">
    <div>
      <div class="container">
        <div class="row">
          <div class="col-sm-2">
            <img :src="sampleImages.sample1" alt="sample1"/>
          </div>
          <div class="col-sm-2">
            <img :src="sampleImages.sample2" alt="sample2"/>
          </div>
          <div class="col-sm-2">
            <img :src="sampleImages.sample3" alt="sample3"/>
          </div>
          <div class="col-sm-2">
            <img :src="sampleImages.sample4" alt="sample4"/>
          </div>
          <div class="col-sm-2">
            <img :src="sampleImages.sample5" alt="sample5"/>
          </div>
          <div class="col-sm-2">
            <img :src="sampleImages.sample6" alt="sample6"/>
          </div>
        </div>
        </br>
        <div class="row">
          <div class="col-sm-4">
            Origin
            </br>
            <img :src="originImage" alt="originImage"/>
            </br>
            <input type="file" @change="onFileSelected">
            <button class="btn btn-success btn-sm" @click="onUpload"> Upload </button>
          </div>

          <div class="col-sm-4">
            Global Style Transfer
            </br>
            <img :src="globalStyleTransferImage" alt="globalStyleTransferImage"/>
          </div>

          <div class="col-sm-4">
            Region-based Style Transfer
            </br>
            <img :src="regionBasedStyleTransferImage" alt="regionBasedStyleTransferImage"/>
          </div>
        </div>
        </br>
        <div class="row">
          <div class="col-sm-2">
            <img :src="styleImages.style1" alt="style1"/>
          </div>
          <div class="col-sm-2">
            <img :src="styleImages.style2" alt="style2"/>
          </div>
          <div class="col-sm-2">
            <img :src="styleImages.style3" alt="style3"/>
          </div>
          <div class="col-sm-2">
            <img :src="styleImages.style4" alt="style4"/>
          </div>
          <div class="col-sm-2">
            <img :src="styleImages.style5" alt="style5"/>
          </div>
          <div class="col-sm-2">
            <img :src="styleImages.style6" alt="style6"/>
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
      //originImage: "https://via.placeholder.com/200",
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
img {
	max-width:100%;
	height:auto;
}

</style>
