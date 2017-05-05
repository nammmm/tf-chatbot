(function(){
  
  var chat = {
    exit: false,
    messageToSend: '',
    init: function() {
      this.cacheDOM();
      this.bindEvents();
    },
    cacheDOM: function() {
      this.$chatHistory = $('.chat-history');
      this.$button = $('button');
      this.$textarea = $('#message-to-send');
      this.$chatHistoryList =  this.$chatHistory.find('ul');
    },
    bindEvents: function() {
      this.$button.on('click', this.addMessage.bind(this));
      this.$textarea.on('keyup', this.addMessageEnter.bind(this));
    },
    updateSend: function() {
      this.scrollToBottom();
      if (this.messageToSend.trim() !== '') {
        var template = Handlebars.compile( $("#message-template").html());
        var context = { 
          messageOutput: this.messageToSend,
          time: this.getCurrentTime()
        };

        this.$chatHistoryList.append(template(context));
        this.scrollToBottom();
        this.$textarea.val('');        
      }
      
    },
    updateResponse: function(response) {
      // responses
      var templateResponse = Handlebars.compile( $("#message-response-template").html());
      var contextResponse = { 
        response: response,
        time: this.getCurrentTime()
      };
      
      setTimeout(function() {
        this.$chatHistoryList.append(templateResponse(contextResponse));
        this.scrollToBottom();
      }.bind(this), 1500);
    },
    addMessage: function() {
      this.messageToSend = this.$textarea.val();
      var msg = this.messageToSend.split("↵").join("")
      if (msg == "quit") {
        this.quit();
      } else {
        this.updateSend();
        if (!this.exit) {
          this.ajax(this);
        }
      }
    },
    addMessageEnter: function(event) {
        // enter was pressed
        if (event.keyCode === 13) {
          this.addMessage();
        }
    },
    scrollToBottom: function() {
      this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
    },
    ajax: function(object) {
      // ajax the JSON to the server
      var msg = object.messageToSend.split("↵").join("")
      $.ajax({
        url: '/receiver',
        data: msg,
        type: 'post',
        success: function (response) {
          object.updateResponse(response);
        }
      });
    },
    quit: function() {
      this.exit = true;
      $.ajax({
        url: '/quit',
        data: "quit",
        type: 'post',
        success: function () {
          this.updateResponse("See you next time!");
          console.log("Quit ...");
        }
      });
    },
    getCurrentTime: function() {
      return new Date().toLocaleTimeString().
              replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
    },
    getRandomItem: function(arr) {
      return arr[Math.floor(Math.random()*arr.length)];
    }
    
  };
  
  chat.init();
  
})();