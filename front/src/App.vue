<template>
  <div class="main-page">
    <div class="left-column">
      <chat-management />
      <button class="button">Выключить навык</button>
    </div>
    <div class="right-column">
      <chat-messages />
      <chat-input />
    </div>
  </div>
</template>

<script setup>
import "@/styles/index.css"
import ChatInput from "@/components/Chat-input.vue"
import ChatManagement from "@/components/Сhat-management.vue"
import ChatMessages from "@/components/Сhat-messages.vue"
</script>

<style lang="scss" scoped>
.main-page {
  margin: 20px auto;
  width: 1000px;
  display: flex;

  .left-column {
    height: 80vh;

    .button {
      width: 100%;
      padding: 16px;
      border-radius: 16px;
      border: 0;
      background: var(--background-color);
      margin-top: 16px;
      box-shadow: 0 4px 6px var(--box-shadow-color);

      &:hover {
        transform: scale(1.01);
        background: var(--main-color);
        color: var(--text-color-white);
      }

      &:active {
        transform: scale(0.99);
      }
    }
  }

  .right-column {
    margin-left: 16px;
  }
}
</style>
