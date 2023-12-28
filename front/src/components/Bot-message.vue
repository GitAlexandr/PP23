<template>
  <div class="message">
    <div v-if="props.content.text" class="message-text">{{ content.text }}</div>
    <a v-if="props.content.file_url" class="message-file-url" :href="content.file_url" target="_blank">{{content.file_url}}</a>
  </div>
</template>

<script setup>
import { defineProps } from "vue";

const props = defineProps({
  content: {
    type: Object
  }
})
</script>

<style scoped lang="scss">
.message {
  overflow: hidden;
  position: relative;
  z-index: 2;
  width: 75%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-radius: 16px;
  border-bottom-left-radius: 0;
  box-shadow: 0 4px 6px var(--box-shadow-color);
  border: 2px solid var(--main-color);
  background: var(--background-color);
}

.message-file-url {
  word-break: break-all; /* или word-break: break-word; */
  color: var(--link-color); /* Замените на цвет вашей ссылки */
  text-decoration: underline; /* Подчеркивание для ссылки */
}

.message-file-url:hover {
  text-decoration: none; /* Убираем подчеркивание при наведении */
}
</style>
