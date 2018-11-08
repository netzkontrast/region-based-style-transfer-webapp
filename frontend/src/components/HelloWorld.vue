<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <input type="file" @change="onFileSelected">
    <button @click="onUpload"> Upload </button>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App',
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
          }
        })
      .then(res=>{
        console.log("uploaded!");
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
