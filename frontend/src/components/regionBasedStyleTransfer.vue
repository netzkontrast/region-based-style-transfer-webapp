<template>
  <div class="RegionBasedStyleTransfer">

  
    <h1>{{ header }}</h1>
    <div>
    <input type="file" @change="onFileSelected">
    <button class="btn btn-success btn-sm" @click="onUpload"> Upload </button>
    </div>
    <img :src="blend_image" alt="blend_image"/>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'RegionBasedStyleTransfer',
  data () {
    return {
      header: 'COMS4731 Region Based Style Transfer',
      selectedFile: null,
      blend_image: "https://via.placeholder.com/300",
    }
  },
  methods: {
    onFileSelected(event){
      console.log("select file");
      this.selectedFile = event.target.files[0];
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

      this.blend_image = "http://localhost:5000/outputs/" + "blend_" + filename + "_wreck" + "." + suffix;
    },
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
