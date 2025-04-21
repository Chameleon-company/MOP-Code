(function(w,d,s){
    var fledgePageUrl = s + '/s/fledge/MSFledgePage.html';
    var fledgePageUETParameter = 'UETEventStr';

	var isEdgeBrowser = /Edg/.test(navigator.userAgent);
	var isOpera = /OPR/.test(navigator.userAgent);
    var isChromeBrowser = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor) && !isEdgeBrowser && !isOpera;

    if(!isEdgeBrowser && !isChromeBrowser) {
       return;
    } 
    
    if (typeof window.CustomEvent !== 'function')
        return;

    d.addEventListener("UetEvent", function(event) {
        var UETParams = getFledgeParams([event.detail.uetEvent]);
        if (UETParams.size > 0) loadIframe(UETParams, fledgePageUrl);
    });

    var rndObjName = d.currentScript.getAttribute('data-ueto');
    if (!rndObjName) return;

    var uetInstance = w[rndObjName];
    if (!uetInstance) return;

    var fledgeParams = getFledgeParams(uetInstance.snippetEventQueue);
    if (fledgeParams.size > 0) loadIframe(fledgeParams, fledgePageUrl);
    
    function getFledgeParams(eventQueue)
    {
        let params = new URLSearchParams();
        eventQueue
            .filter(function(evtStr) { 
                var kvps = new URLSearchParams(evtStr);
                if (!kvps.get('cdl')) return false;
                if (!kvps.get('cdl').includes('control_2') && !kvps.get('cdl').includes('treatment') && isChromeBrowser) return false;
                if (!kvps.get('cdl').includes('treatment_1.1') && isEdgeBrowser) return false;
                if (kvps.has('gdpr','Y') && kvps.has('tcf','st=L') && kvps.has('as','D')) return false;
                if (kvps.has('asc','D')) return false;
                return kvps.get('evt') !== 'custom';
                })
            .forEach((uetEventStr) => 
                {
                    params.append(fledgePageUETParameter, encodeURIComponent(uetEventStr));
                });
        return params;
    }
    
    function loadIframe(params, fledgePageUrl){
        var fledgeIframe = document.createElement('iframe');
        fledgeIframe.style.display = "none";
        fledgeIframe.id = 'FledgeIFrame' + makeRandomStr(10);
        document.body.appendChild(fledgeIframe);
        
        fledgeIframe.src = fledgePageUrl + '?' + params.toString();
    }
    
    function makeRandomStr(length) {
      var str = '';
      for (var i = 0; i < Math.ceil(length / 2); i++)
            str += _SB(Math.floor(Math.random() * 256));
        return str.slice(-length);
    };
    
    function _SB(n) {
        // indexEnd is insurance to make sure no more than 2 chars
        return (n + 0x100).toString(16).substring(1, 3);
    };
})(window, document,"https://bat.bing.com");
(function(w,d,c,k,a,b,t,e,h) {
    var cs = d.currentScript;
    if (cs) {
        var uo = cs.getAttribute('data-ueto');
        if (uo && w[uo] && typeof w[uo].setUserSignals === 'function') {
            w[uo].setUserSignals({'ea': c, 'kc': k, 'at': a, 'bi': b, 'pt': t, 'ec': e, 'ah': h});
        }
    }
})(window, document, false, false, true, false, false, false, false);
