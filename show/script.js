// show/script.js

// เพิ่มการตรวจสอบฟอร์มก่อนส่ง
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    form.addEventListener("submit", function (event) {
      // ตรวจสอบว่าฟิลด์ทั้งหมดถูกกรอกหรือไม่
      const inputs = form.querySelectorAll('input[type="text"]');
      let valid = true;
      inputs.forEach(function (input) {
        if (input.value.trim() === "") {
          valid = false;
          input.style.border = "1px solid #ff5252";
        } else {
          input.style.border = "none";
        }
      });
  
      if (!valid) {
        event.preventDefault();
        alert("กรุณากรอกข้อมูลให้ครบถ้วน");
      }
    });
  });
  
  // เพิ่มการทำงานของปุ่มแสดง/ซ่อนตาราง
  document.addEventListener("DOMContentLoaded", function () {
    const toggleTableBtn = document.getElementById("toggle-table-btn");
    const tableContainer = document.getElementById("table-container");
  
    toggleTableBtn.addEventListener("click", function () {
      if (tableContainer.style.display === "none") {
        tableContainer.style.display = "block";
        toggleTableBtn.textContent = "ซ่อนข้อมูลตาราง";
      } else {
        tableContainer.style.display = "none";
        toggleTableBtn.textContent = "แสดงข้อมูลตาราง";
      }
    });
  });
  