<template>
  <div class="RegionBasedStyleTransfer">
    <h1>{{ header }}</h1>
    <input type="file" @change="onFileSelected">
    <button @click="onUpload"> Upload </button>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'RegionBasedStyleTransfer',
  data () {
    return {
      header: 'COMS4731 Region Based Style Transfer',
      selectedFile: null
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
      });
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
