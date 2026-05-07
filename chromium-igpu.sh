#!/bin/bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  chromium-browser \
  --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE \
  --ozone-platform=x11
