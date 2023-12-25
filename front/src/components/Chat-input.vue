<template>
  <div class="container" @click="focusInput">
    <input class="input" :placeholder="placeholder" v-model="textInput" @keyup.enter="pushMessage">
    <button class="input-push" @click="pushMessage">
      <svg-icon class="mdi" type="mdi" :path="mdiArrowRightCircle" />
    </button>
    <voice-recording />
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import SvgIcon from '@jamescoyle/vue-icon';
import { mdiArrowRightCircle } from '@mdi/js';
import store from "@/store/index.js";
import VoiceRecording from "@/components/Voice-recording.vue";

const placeholder = ref('Введите вопрос');
const textInput = ref("");
const isInputFocused = ref(false);

function pushMessage() {
  if (textInput.value.length !== 0) {
    store.dispatch('sendMessageToChat', textInput.value);
    textInput.value = "";
  }
}

function focusInput() {
  isInputFocused.value = true;
  const input = document.querySelector(".input");
  if (input) {
    input.focus();
  }
}

onMounted(() => {
  focusInput();
});
</script>

<style scoped lang="scss">
.container {
  margin-top: 16px;
  padding: 12px;
  border-radius: 16px;
  background: var(--background-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 6px var(--box-shadow-color);
}

.input {
  margin-left: 4px;
  width: 100%;
  background: var(--background-color);
  border: 0;
  font-size: 16px;

  &:focus {
    outline: none;
  }
}

.input-push {
  width: 28px;
  height: 28px;
  background: var(--background-color);
  border: 0;
  color: var(--text-color-dark-grey);
}
</style>
