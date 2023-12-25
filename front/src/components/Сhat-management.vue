<template>
  <div class="container">
    <img :src="ava" alt="аватарка">
    <div class="buttons">
      <div>
        <input class="hide" type="file" ref="fileInput" @change="handleFileChange"/>
        <button class="button file-input" @click="uploadFile">Загрузить файл</button>
      </div>
      <button class="button" @click="clearMessages">Очистить</button>
      <list-of-skills/>
    </div>
  </div>
</template>

<script setup>
import ava from "@/assets/ava.jpg"
import {useStore} from "vuex"
import {ref} from 'vue'
import ListOfSkills from "@/components/List-of-skills.vue"

const store = useStore()

function clearMessages() {
  store.commit("clearMessage")
}

const fileInput = ref(null)

const handleFileChange = () => {
  const file = fileInput.value.files[0];

  if (file) {
    const reader = new FileReader();

    reader.onload = (event) => {
      const fileContent = event.target.result;

      const fileInput = {
        name: file.name,
        size: file.size,
        content: fileContent,
      }

      store.dispatch('sendAMessageFile', fileInput)
      store.dispatch('chatHistory');
    };

    reader.readAsDataURL(file);
  }
};

const uploadFile = () => {
  fileInput.value.click()
};
</script>

<style scoped lang="scss">
.container {
  width: 220px;
  padding: 16px;
  border-radius: 16px;
  background: var(--background-color);
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 6px var(--box-shadow-color);

  img {
    margin: 0 auto;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    border: 4px solid var(--main-color);
    box-shadow: 0 4px 6px var(--box-shadow-color);
  }

  .file-input {
    width: 100%;
  }

  .buttons {
    margin-top: 20px;
    display: flex;
    flex-direction: column;

    .button {
      margin: 4px 0;
      padding: 12px;
      background: var(--main-color);
      color: var(--text-color-white);
      border: 0;
      border-radius: 16px;
      font-size: 16px;
      box-shadow: 0 4px 6px var(--box-shadow-color);

      &:hover {
        transform: scale(1.01);
      }

      &:active {
        transform: scale(0.99);
      }
    }
  }
}
</style>
