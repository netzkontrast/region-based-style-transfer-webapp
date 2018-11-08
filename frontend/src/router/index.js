import Vue from 'vue'
import Router from 'vue-router'
import RegionBasedStyleTransfer from '@/components/regionBasedStyleTransfer'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'regionBasedStyleTransfer',
      component: RegionBasedStyleTransfer
    }
  ]
})
