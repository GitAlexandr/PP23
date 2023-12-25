<template>
  <div class="wrapper">
    <img class="logo" :src="OCRVLogo" alt="Логотип компании ОЦРВ">
    <div class="container">
      <div class="content">
        <div class="message" v-for="({ author, content }, index) in messages" :key="index">
          <user-message v-if="author === 'user'" :content="content" />
          <bot-message v-if="author === 'bot'" :content="content" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import UserMessage from "@/components/User-message.vue";
import BotMessage from "@/components/Bot-message.vue";
import { computed } from "vue";
import { useStore } from "vuex";
import OCRVLogo from "@/assets/OCRV-Logo.svg"

const store = useStore();
const messages = computed(() => store.getters.getMessages);
</script>

<style scoped lang="scss">
.wrapper {
  position: relative;
  z-index: 1;

  .logo {
    top: 30%;
    left: 0;
    position: absolute;
    width: auto;
    opacity: 0.3;
    z-index: 2;
  }

  .container {
    height: 80vh;
    width: 700px;
    padding: 16px;
    border-radius: 16px;
    background: var(--background-color);
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 6px var(--box-shadow-color);
    overflow: hidden;
    overflow-y: auto;

    .content {
      margin-top: auto;
      width: auto;

      .message {
        margin-bottom: 4px;
      }
    }

    &::-webkit-scrollbar {
      width: 8px;
    }

    &::-webkit-scrollbar-track {
      border-radius: 4px;
      background-color: transparent;
      border: 0;
    }

    &::-webkit-scrollbar-thumb {
      border-radius: 4px;
      background: var(--main-color);
    }
  }
}
</style>
